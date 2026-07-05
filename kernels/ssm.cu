// Nova Core CUDA Kernels
// Compiled to PTX at build time via build.rs
// Compatible with NVIDIA GPUs: T4 (sm_75), A100 (sm_80), RTX 30xx (sm_86), RTX 40xx (sm_89), H100 (sm_90)

// ============================================================================
// Helper: warp-level reduction for max/min operations
// ============================================================================
__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// ============================================================================
// SSM Selective Scan - O(n) state space model step
// Input:  A (d_state), B (d_inner), C (d_state), h (d_state)
//         x (d_inner), delta (d_inner), delta_bias (d_inner), D (d_inner)
// Output: y (d_inner)
// Each thread processes one feature dimension
// ============================================================================
extern "C" __global__ void selective_scan(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    float* __restrict__ h,
    const float* __restrict__ x,
    const float* __restrict__ delta,
    const float* __restrict__ delta_bias,
    const float* __restrict__ D,
    float* __restrict__ output,
    int d_inner,
    int d_state
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_inner) return;

    float dt = delta[idx] + delta_bias[idx];
    dt = log1pf(expf(dt)); // softplus

    float x_dt = x[idx] * dt;

    // State space update: h = A * h + B * x_dt
    float h_new = 0.0f;
    for (int s = 0; s < d_state; s++) {
        int s_idx = s * d_inner + idx;
        float a_val = A[s_idx];
        float b_val = B[s_idx];
        h_new += a_val * h[s_idx] + b_val * x_dt;
    }

    // Write back h
    for (int s = 0; s < d_state; s++) {
        int s_idx = s * d_inner + idx;
        h[s_idx] = h_new / (float)d_state;
    }

    // Output: y = C * h + D * x
    float c_dot_h = 0.0f;
    for (int s = 0; s < d_state; s++) {
        c_dot_h += C[s * d_inner + idx] * h_new;
    }
    output[idx] = c_dot_h / (float)d_state + D[idx] * x[idx];
}

// ============================================================================
// SSM Transform Batch - Process multiple pulses through SSM
// Input:  A (d_state*d_inner), B (d_state*d_inner), C (d_state*d_inner)
//         h (d_state*d_inner), delta (d_inner), delta_bias (d_inner)
//         D (d_inner), pulses (num_pulses * d_inner)
// Output: output (num_pulses * d_inner)
// ============================================================================
extern "C" __global__ void ssm_transform_batch(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    float* __restrict__ h,
    const float* __restrict__ delta,
    const float* __restrict__ delta_bias,
    const float* __restrict__ D,
    const float* __restrict__ pulses,
    float* __restrict__ output,
    int num_pulses,
    int d_inner,
    int d_state
) {
    int pulse_idx = blockIdx.x;
    int feature_idx = threadIdx.x;
    
    if (pulse_idx >= num_pulses || feature_idx >= d_inner) return;
    
    int idx = pulse_idx * d_inner + feature_idx;
    float val = pulses[idx];
    
    float dt = delta[feature_idx] + delta_bias[feature_idx];
    dt = log1pf(expf(dt)); // softplus
    
    float x_dt = val * dt;
    
    // State update
    float h_new = 0.0f;
    for (int s = 0; s < d_state; s++) {
        h_new += A[s * d_inner + feature_idx] * h[s * d_inner + feature_idx] 
               + B[s * d_inner + feature_idx] * x_dt;
    }
    
    for (int s = 0; s < d_state; s++) {
        h[s * d_inner + feature_idx] = h_new / (float)d_state;
    }
    
    // Output
    float c_dot_h = 0.0f;
    for (int s = 0; s < d_state; s++) {
        c_dot_h += C[s * d_inner + feature_idx] * h_new;
    }
    output[idx] = c_dot_h / (float)d_state + D[feature_idx] * val;
}

// ============================================================================
// Field Update - Gradient-based field evolution
// pulses_content (num_pulses * dim), pulses_weight (num_pulses), 
// field_state (dim), field_momentum (dim)
// ============================================================================
extern "C" __global__ void field_update(
    const float* __restrict__ pulses_content,
    const float* __restrict__ pulses_weight,
    float* __restrict__ field_state,
    float* __restrict__ field_momentum,
    float learning_rate,
    float diffusion,
    int num_pulses,
    int dim
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;

    float grad = 0.0f;
    float total_weight = 0.0f;
    
    for (int p = 0; p < num_pulses; p++) {
        float w = pulses_weight[p];
        grad += w * pulses_content[p * dim + d];
        total_weight += w;
    }
    
    if (total_weight > 0.0f) {
        grad /= total_weight;
        grad -= field_state[d] * diffusion;
        
        // Momentum update
        float momentum = field_momentum[d] * 0.9f + grad * learning_rate;
        field_momentum[d] = momentum;
        field_state[d] += momentum;
    }
}

// ============================================================================
// Field Diffuse - Apply diffusion to field state
// ============================================================================
extern "C" __global__ void field_diffuse(
    const float* __restrict__ pulses_content,
    float* __restrict__ field_state,
    float diffusion_factor,
    int num_pulses,
    int dim
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    
    // Diffusion towards average activation
    float avg = 0.0f;
    for (int p = 0; p < num_pulses; p++) {
        avg += pulses_content[p * dim + d];
    }
    avg /= max(num_pulses, 1);
    
    field_state[d] += (avg - field_state[d]) * diffusion_factor;
}

// ============================================================================
// Cosine Similarity - Find closest vocabulary word to query
// query (dim), vocabulary (vocab_size * dim), vocab_norms (vocab_size)
// Output: similarities (vocab_size)
// ============================================================================
extern "C" __global__ void cosine_similarity(
    const float* __restrict__ query,
    const float* __restrict__ vocabulary,
    const float* __restrict__ vocab_norms,
    float* __restrict__ similarities,
    int vocab_size,
    int dim
) {
    int word_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (word_idx >= vocab_size) return;

    float dot = 0.0f;
    for (int d = 0; d < dim; d++) {
        dot += query[d] * vocabulary[word_idx * dim + d];
    }
    
    float query_norm = 0.0f;
    for (int d = 0; d < dim; d++) {
        query_norm += query[d] * query[d];
    }
    query_norm = sqrtf(max(query_norm, 1e-10f));
    
    similarities[word_idx] = dot / (query_norm * max(vocab_norms[word_idx], 1e-10f));
}

// ============================================================================
// Vector Add: a = a + b * scale1 + scale2
// ============================================================================
extern "C" __global__ void vector_add(
    float* __restrict__ a,
    const float* __restrict__ b,
    float scale_a,
    float scale_b,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = a[i] * scale_a + b[i] * scale_b;
}

// ============================================================================
// Vector Clamp: a[i] = clamp(a[i], min_val, max_val)
// ============================================================================
extern "C" __global__ void vector_clamp(
    float* a,
    float min_val,
    float max_val,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = fminf(fmaxf(a[i], min_val), max_val);
}

// ============================================================================
// Core Process - Single core processing of pulses
// Each block processes one core's memory and SSM state
// ============================================================================
extern "C" __global__ void core_process(
    float* __restrict__ pulses_content,
    float* __restrict__ pulses_entropy,
    float* __restrict__ pulses_weight,
    const float* __restrict__ core_memory,
    const float* __restrict__ core_internal_state,
    float core_gate,
    const float* __restrict__ ssm_a,
    const float* __restrict__ ssm_b,
    const float* __restrict__ ssm_c,
    float* __restrict__ ssm_h,
    const float* __restrict__ ssm_delta,
    const float* __restrict__ ssm_delta_bias,
    const float* __restrict__ ssm_d,
    int num_pulses,
    int dim,
    int num_cores,
    int memory_size,
    int d_state
) {
    int pulse_idx = blockIdx.x;
    if (pulse_idx >= num_pulses) return;
    
    int d = threadIdx.x;
    if (d >= dim) return;
    
    int idx = pulse_idx * dim + d;
    float val = pulses_content[idx];
    
    // Apply memory influence (content-addressable memory)
    float mem_influence = 0.0f;
    for (int m = 0; m < min(memory_size, 256); m++) {
        mem_influence += core_memory[m * dim + d] * val;
    }
    mem_influence = tanhf(mem_influence * 0.01f);
    
    // Apply SSM transformation
    float dt = ssm_delta[d] + ssm_delta_bias[d];
    dt = log1pf(expf(dt));
    float ssm_out = 0.0f;
    for (int s = 0; s < d_state; s++) {
        ssm_out += ssm_c[s * dim + d] * ssm_h[s * dim + d];
    }
    
    // Blend: gate controls core influence
    float gate = core_gate;
    float new_val = val * (1.0f - gate) 
                  + (mem_influence * 0.3f + ssm_out * 0.5f + val * 0.2f) * gate;
    
    // Update SSM state
    for (int s = 0; s < d_state; s++) {
        ssm_h[s * dim + d] = ssm_h[s * dim + d] * 0.5f + val * ssm_b[s * dim + d] * 0.5f;
    }
    
    // Update entropy based on activation change
    float entropy_change = fabsf(new_val - val);
    pulses_entropy[idx] = pulses_entropy[idx] * 0.9f + entropy_change * 0.1f;
    
    // Update weight
    pulses_weight[pulse_idx] = fminf(pulses_weight[pulse_idx] + 0.01f, 1.0f);
    
    pulses_content[idx] = fminf(fmaxf(new_val, -1.0f), 1.0f);
}