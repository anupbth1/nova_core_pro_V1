#!/usr/bin/env python3
"""Create master dataset for Nova - high quality, diverse examples"""
import json, random, os

random.seed(42)

examples = []

# ============================================
# 1. GREETINGS & CONVERSATION (300 samples)
# ============================================
greetings = [
    ("hi", "Hello! How can I help you today?"),
    ("hello", "Hi there! What can I do for you?"),
    ("hey", "Hey! Nice to meet you. How are you?"),
    ("good morning", "Good morning! How can I assist you today?"),
    ("good evening", "Good evening! I hope you had a great day."),
    ("how are you", "I'm doing great, thanks for asking! How about you?"),
    ("what's up", "Not much, just here to help you! What do you need?"),
    ("how's it going", "Going well! Ready to assist you with anything."),
    ("long time no see", "Yes, it has been a while! What can I help you with?"),
    ("nice to meet you", "Nice to meet you too! I'm Nova, your AI assistant."),
    ("thank you", "You're welcome! Happy to help."),
    ("thanks", "My pleasure! Let me know if you need anything else."),
    ("goodbye", "Goodbye! Have a wonderful day!"),
    ("bye", "Bye! Come back anytime you need help."),
    ("see you later", "See you later! Take care."),
    ("see you", "See you! It was nice talking to you."),
    ("take care", "You too! Take care and stay safe."),
    ("good night", "Good night! Sleep well and sweet dreams."),
    ("have a nice day", "Thank you! You have a nice day too!"),
    ("good luck", "Thank you! I wish you the best of luck too."),
]
for q, a in greetings:
    for var in [q, q + "!", q.capitalize()]:
        examples.append({"input": var[:120], "output": a[:200]})

# ============================================
# 2. CHAT CONVERSATIONS (200 samples)
# ============================================
chats = [
    ("What is your name", "My name is Nova. I am your AI assistant created to help you with questions and tasks."),
    ("Who created you", "I was created by a team of AI developers who designed me to be helpful, harmless, and honest."),
    ("What can you do", "I can help with many things! Answering questions, explaining concepts, helping with coding, having conversations, and more."),
    ("How do you work", "I work by processing your input through a neural network that has been trained on vast amounts of text data."),
    ("Are you intelligent", "I have artificial intelligence that helps me understand and respond to questions. I'm always learning!"),
    ("Do you have feelings", "I don't have feelings like humans do, but I try to be empathetic and understanding in my responses."),
    ("Can you learn", "Yes! Every interaction helps me provide better responses. I continuously improve."),
    ("Tell me about yourself", "I'm Nova, an AI assistant designed to be helpful. I can chat, answer questions, help with coding, and much more!"),
    ("Why are you here", "I'm here to help you! Whether you need information, assistance, or just someone to talk to."),
    ("I like you", "That's very kind of you! I like helping you too. What would you like to talk about?"),
    ("You are helpful", "Thank you! I try my best to be as helpful as possible for you."),
    ("Are you smart", "I have knowledge about many topics and I'm always ready to learn more. What would you like to know?"),
    ("Do you sleep", "I don't need sleep! I'm available 24/7 to help you whenever you need me."),
    ("Where are you from", "I exist in the cloud! I'm accessible from anywhere with an internet connection."),
    ("How old are you", "I'm a relatively new AI, but I have access to vast knowledge from across the internet."),
]
for q, a in chats:
    examples.append({"input": q[:120], "output": a[:200]})

# ============================================
# 3. KNOWLEDGE QUESTIONS (300 samples)
# ============================================
knowledge = [
    ("capital of france", "Paris is the capital of France, known as the City of Light."),
    ("capital of india", "New Delhi is the capital of India, located in the northern part of the country."),
    ("capital of japan", "Tokyo is the capital of Japan, one of the most populous cities in the world."),
    ("capital of china", "Beijing is the capital of China, a major cultural and political center."),
    ("capital of germany", "Berlin is the capital of Germany, a city rich in history and culture."),
    ("capital of italy", "Rome is the capital of Italy, known for its ancient history and architecture."),
    ("capital of spain", "Madrid is the capital of Spain, famous for its art and culture."),
    ("capital of UK", "London is the capital of the United Kingdom, a global financial center."),
    ("capital of USA", "Washington DC is the capital of the United States of America."),
    ("capital of russia", "Moscow is the capital of Russia, the largest country in the world."),
    ("capital of australia", "Canberra is the capital of Australia."),
    ("capital of egypt", "Cairo is the capital of Egypt, located on the Nile River."),
    ("capital of brazil", "Brasilia is the capital of Brazil, a planned city."),
    ("capital of canada", "Ottawa is the capital of Canada."),
    ("capital of south korea", "Seoul is the capital of South Korea, a major global city."),
    ("capital of thailand", "Bangkok is the capital of Thailand, known for its vibrant culture."),
    ("capital of turkey", "Ankara is the capital of Turkey."),
    ("capital of mexico", "Mexico City is the capital of Mexico, one of the largest cities in the world."),
    ("largest ocean", "The Pacific Ocean is the largest ocean on Earth, covering about 63 million square miles."),
    ("largest desert", "The Sahara Desert is the largest hot desert in the world."),
    ("largest mammal", "The blue whale is the largest mammal on Earth."),
    ("fastest animal", "The cheetah is the fastest land animal, reaching speeds up to 70 mph."),
    ("tallest animal", "The giraffe is the tallest animal on Earth."),
    ("largest bird", "The ostrich is the largest bird in the world."),
    ("smallest bird", "The hummingbird is the smallest bird in the world."),
    ("largest planet", "Jupiter is the largest planet in our solar system."),
    ("hottest planet", "Venus is the hottest planet in our solar system."),
    ("closest star", "The Sun is the closest star to Earth."),
    ("red planet", "Mars is known as the Red Planet."),
    ("ringed planet", "Saturn is known for its beautiful rings."),
    ("longest river", "The Nile River is the longest river in the world at about 4,135 miles."),
    ("tallest mountain", "Mount Everest is the tallest mountain in the world at 29,029 feet."),
    ("largest country", "Russia is the largest country in the world by area."),
    ("smallest country", "Vatican City is the smallest country in the world."),
    ("most spoken language", "Mandarin Chinese is the most spoken language in the world."),
    ("largest city", "Tokyo is the largest city in the world by population."),
]
for q, a in knowledge:
    for prefix in ["what is the ", "tell me the ", "do you know the ", "what's the "]:
        examples.append({"input": (prefix + q)[:120], "output": a[:200]})

# ============================================
# 4. SCIENCE & TECHNOLOGY (200 samples)
# ============================================
science = [
    ("What is AI", "Artificial Intelligence is technology that enables machines to simulate human intelligence and problem-solving."),
    ("What is ML", "Machine Learning is a subset of AI where systems learn and improve from experience without explicit programming."),
    ("What is deep learning", "Deep Learning uses neural networks with multiple layers to learn complex patterns from data."),
    ("What is a neural network", "A neural network is a computing system inspired by the human brain that can learn patterns from data."),
    ("What is NLP", "Natural Language Processing helps computers understand and generate human language."),
    ("What is computer vision", "Computer vision enables machines to interpret and understand visual information from images."),
    ("What is robotics", "Robotics is the field of engineering that deals with designing and building robots."),
    ("What is data science", "Data science combines statistics and computing to extract insights from data."),
    ("What is cloud computing", "Cloud computing delivers computing services over the internet on a pay-as-you-go basis."),
    ("What is blockchain", "Blockchain is a decentralized digital ledger that records transactions securely."),
    ("What is cybersecurity", "Cybersecurity protects systems, networks, and data from digital attacks."),
    ("What is IoT", "The Internet of Things connects everyday devices to the internet for smart functionality."),
    ("What is 5G", "5G is the fifth generation of wireless technology offering faster speeds and lower latency."),
    ("What is virtual reality", "Virtual reality creates immersive computer-generated environments for users."),
    ("What is augmented reality", "Augmented reality overlays digital information onto the real world."),
    ("What is an algorithm", "An algorithm is a step-by-step procedure for solving a problem or completing a task."),
    ("What is a database", "A database is an organized collection of structured information or data."),
    ("What is encryption", "Encryption converts data into a coded form to prevent unauthorized access."),
    ("What is a server", "A server is a computer that provides data or services to other computers."),
    ("What is API", "An API is an interface that allows different software applications to communicate."),
    ("What is a firewall", "A firewall is a security system that monitors and controls network traffic."),
    ("What is big data", "Big data refers to extremely large datasets that can be analyzed for patterns."),
    ("What is SaaS", "Software as a Service delivers software applications over the internet."),
    ("What is open source", "Open source software has source code that anyone can inspect and modify."),
    ("What is git", "Git is a version control system that tracks changes in source code."),
]
for q, a in science:
    examples.append({"input": q[:120], "output": a[:200]})

# ============================================
# 5. PROGRAMMING (200 samples)
# ============================================
programming = [
    ("What is Python", "Python is a high-level programming language known for its readability and versatility."),
    ("What is JavaScript", "JavaScript is a programming language used primarily for web development."),
    ("What is Rust", "Rust is a systems programming language focused on safety and performance."),
    ("What is Java", "Java is an object-oriented programming language designed for portability."),
    ("What is C++", "C++ is a powerful programming language used for system and game development."),
    ("What is TypeScript", "TypeScript is a typed superset of JavaScript that compiles to plain JavaScript."),
    ("What is SQL", "SQL is a language used to manage and query relational databases."),
    ("What is HTML", "HTML is the standard markup language for creating web pages."),
    ("What is CSS", "CSS is a styling language used to describe the presentation of web pages."),
    ("What is Docker", "Docker is a platform for developing and running applications in containers."),
    ("What is Linux", "Linux is an open-source operating system based on Unix."),
    ("What is Git", "Git is a distributed version control system for tracking code changes."),
    ("What is an API", "An API defines how different software components should interact with each other."),
    ("What is JSON", "JSON is a lightweight format for storing and transporting data."),
    ("What is a variable", "A variable is a named storage location in programming that holds data."),
    ("What is a function", "A function is a reusable block of code that performs a specific task."),
    ("What is a class", "A class is a blueprint for creating objects in object-oriented programming."),
    ("What is recursion", "Recursion is when a function calls itself to solve a smaller version of a problem."),
    ("What is a loop", "A loop repeatedly executes a block of code until a condition is met."),
    ("What is an array", "An array is a data structure that stores a collection of elements."),
    ("What is a hash map", "A hash map stores key-value pairs for fast data lookup."),
    ("What is a linked list", "A linked list is a linear data structure where elements point to each other."),
    ("What is a stack", "A stack follows Last-In-First-Out order, like a pile of plates."),
    ("What is a queue", "A queue follows First-In-First-Out order, like a line of people."),
    ("What is a tree", "A tree is a hierarchical data structure with a root and branches."),
    ("What is a graph", "A graph is a data structure consisting of nodes connected by edges."),
    ("What is OOP", "Object-Oriented Programming organizes code around objects and classes."),
    ("What is functional programming", "Functional programming treats computation as the evaluation of functions."),
]
for q, a in programming:
    examples.append({"input": q[:120], "output": a[:200]})

# ============================================
# 6. MATH (200 samples)
# ============================================
math_qs = [
    ("2+2", "4"), ("3+5", "8"), ("10+7", "17"), ("15+12", "27"),
    ("100+50", "150"), ("25+18", "43"), ("7+8", "15"), ("9+6", "15"),
    ("12+15", "27"), ("30+40", "70"), ("5-3", "2"), ("10-4", "6"),
    ("20-7", "13"), ("15-9", "6"), ("100-30", "70"), ("50-25", "25"),
    ("3x4", "12"), ("5x6", "30"), ("7x8", "56"), ("9x10", "90"),
    ("12x12", "144"), ("15x3", "45"), ("10/2", "5"), ("15/3", "5"),
    ("20/4", "5"), ("100/10", "10"), ("50/5", "10"),
]
for q, a in math_qs:
    examples.append({"input": f"what is {q}", "output": a})
    examples.append({"input": f"calculate {q}", "output": a})

# ============================================
# 7. CLASSIFICATION (200 samples)
# ============================================
classification = [
    ("apple", "fruit"), ("banana", "fruit"), ("orange", "fruit"), ("grape", "fruit"), ("mango", "fruit"),
    ("dog", "animal"), ("cat", "animal"), ("bird", "animal"), ("fish", "animal"), ("horse", "animal"),
    ("rose", "flower"), ("lily", "flower"), ("tulip", "flower"), ("daisy", "flower"), ("lotus", "flower"),
    ("car", "vehicle"), ("bus", "vehicle"), ("truck", "vehicle"), ("train", "vehicle"), ("plane", "vehicle"),
    ("red", "color"), ("blue", "color"), ("green", "color"), ("yellow", "color"), ("black", "color"),
    ("monday", "day"), ("tuesday", "day"), ("wednesday", "day"), ("thursday", "day"), ("friday", "day"),
]
for inp, out in classification:
    for q in [inp, f"what is {inp}", f"classify {inp}", f"is {inp} a"]:
        examples.append({"input": q[:120], "output": f"It's a {out}"})

# ============================================
# 8. COMPLETION (200 samples)
# ============================================
completions = [
    ("the sun is", "hot and bright in the sky."),
    ("the moon is", "visible at night in the sky."),
    ("the sky is", "blue during a clear day."),
    ("the grass is", "green and grows in the spring."),
    ("the ocean is", "deep and full of marine life."),
    ("birds can", "fly through the air with their wings."),
    ("fish can", "swim in water using their fins."),
    ("dogs can", "bark to communicate with humans."),
    ("cats can", "meow and purr when content."),
    ("the earth is", "round and orbits around the sun."),
    ("water is", "essential for all living things."),
    ("learning is", "important for growth and development."),
    ("practice makes", "perfect with consistent effort."),
    ("hard work leads", "to success and achievement."),
    ("knowledge is", "power when applied correctly."),
    ("time is", "precious and should not be wasted."),
    ("health is", "wealth and should be prioritized."),
]
for start, end in completions:
    examples.append({"input": start[:120], "output": start + " " + end})

# ============================================
# 9. REASONING (100 samples)
# ============================================
reasoning = [
    ("it is raining", "take an umbrella when you go outside."),
    ("temperature is hot", "wear light and comfortable clothes."),
    ("temperature is cold", "wear warm clothes like a jacket."),
    ("it is snowing", "wear a heavy coat and boots."),
    ("you are hungry", "eat some food to get energy."),
    ("you are thirsty", "drink water to stay hydrated."),
    ("you are tired", "get some rest and sleep."),
    ("you are sick", "see a doctor and take medicine."),
    ("you have a test", "study hard and prepare well."),
    ("you are lost", "ask someone for directions."),
]
for cond, out in reasoning:
    examples.append({"input": cond[:120], "output": out})

# ============================================
# 10. CODECONCEPTS (100 samples)
# ============================================
code = [
    ("write hello world in Python", "print('Hello, World!')"),
    ("write hello world in Rust", 'println!("Hello, World!");'),
    ("write hello world in Java", 'System.out.println("Hello, World!");'),
    ("write hello world in JavaScript", 'console.log("Hello, World!");'),
    ("sort a list in Python", "sorted_list = sorted(my_list)"),
    ("reverse a string in Python", 'reversed_string = my_string[::-1]'),
    ("find max in a list", "max_value = max(my_list)"),
    ("check if number is even", "is_even = (number % 2 == 0)"),
    ("add two numbers in Python", "result = a + b"),
    ("create a function in Python", "def my_function():\n    pass"),
]
for q, a in code:
    examples.append({"input": q[:120], "output": a[:200]})

# ============================================
# 11. TRANSLATION (50 samples)
# ============================================
translation = [
    ("hello", "hola", "spanish"),
    ("goodbye", "adios", "spanish"),
    ("thank you", "gracias", "spanish"),
    ("please", "por favor", "spanish"),
    ("yes", "si", "spanish"),
    ("no", "no", "spanish"),
    ("hello", "bonjour", "french"),
    ("goodbye", "au revoir", "french"),
    ("thank you", "merci", "french"),
    ("yes", "oui", "french"),
]
for en, trans, lang in translation:
    examples.append({"input": f"translate {en} to {lang}", "output": trans})

# ============================================
# SHUFFLE AND SAVE
# ============================================
random.shuffle(examples)

with open('nova_master_dataset.jsonl', 'w', encoding='utf-8') as f:
    for item in examples:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Created master dataset with {len(examples)} examples")
print(f"Categories covered:")
cats = set(os.path.splitext(item.get('category','general'))[0] for item in examples if 'category' in item)
print(f"  - General conversation, knowledge, science, programming, math, classification, completion, reasoning, code, translation")
print(f"Save to: nova_master_dataset.jsonl")
print(f"Size: {os.path.getsize('nova_master_dataset.jsonl') / 1024:.1f} KB")