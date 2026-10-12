import re
import numpy as np
from flask import Flask, jsonify, render_template, request
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# -------------------------------------------------------------
# 12 BENCHMARK CAREER TRACKS & DOMAIN DATA
# -------------------------------------------------------------
JOB_TAXONOMY = {
    "AI / Machine Learning Engineer": {
        "salary_inr": "₹8.5 - ₹18.0 LPA",
        "salary_usd": "$115,000 - $165,000",
        "demand": "Critical Demand",
        "required_skills": [
            "Python", "Machine Learning", "Deep Learning", 
            "Neural Networks", "PyTorch & TensorFlow", "Data Preprocessing", "Git / GitHub"
        ],
        "interview_topics": [
            "Explain bias-variance tradeoff and entropy pruning in Decision Trees.",
            "How do you resolve vanishing gradients in deep backpropagation?",
            "Techniques for handling high class-imbalance in production datasets.",
            "Structuring end-to-end ML training pipelines with Scikit-Learn."
        ],
        "companies": [
            {"name": "Bosch Global Software Technologies", "city": "Coimbatore", "hub": "CHIL SEZ / Saravanampatti", "type": "Automotive AI", "openings": 14},
            {"name": "IBM India Software Labs", "city": "Bengaluru", "hub": "Embassy GolfLinks", "type": "Enterprise AI", "openings": 18},
            {"name": "Zoho Corporation", "city": "Chennai", "hub": "Estancia IT Park", "type": "SaaS Platform AI", "openings": 12},
            {"name": "Tata Consultancy Services (TCS AI Labs)", "city": "Coimbatore", "hub": "TIDEL Park", "type": "Cognitive Solutions", "openings": 15},
            {"name": "LatentView Analytics", "city": "Chennai", "hub": "Ramanujan IT City", "type": "Applied Intelligence", "openings": 8}
        ],
        "ibm_courses": [
            {"title": "Artificial Intelligence Fundamentals", "badge": "IBM SkillsBuild Credly Badge", "duration": "10 Hours", "url": "https://skills.yourlearning.ibm.com/activity/PLAN-CD7DED2B108B"},
            {"title": "Machine Learning with Python & Watson", "badge": "IBM Watson Certification", "duration": "14 Hours", "url": "https://skillsbuild.org/adult-learners/explore-learning/artificial-intelligence"},
            {"title": "Deep Learning Ecosystem & Neural Networks", "badge": "Applied AI Digital Credential", "duration": "12 Hours", "url": "https://www.credly.com/organizations/ibm/badges"},
            {"title": "AI Ethics, Governance & Model Explainability", "badge": "Trustworthy AI Badge", "duration": "8 Hours", "url": "https://skillsbuild.org/adult-learners/explore-learning"}
        ]
    },
    "GenAI & LLM Solutions Engineer": {
        "salary_inr": "₹10.0 - ₹22.0 LPA",
        "salary_usd": "$130,000 - $185,000",
        "demand": "Exponential Growth",
        "required_skills": [
            "Python", "Prompt Engineering", "Deep Learning", 
            "LangChain & Vector DBs", "FastAPI & Backend", "Neural Networks", "Git / GitHub"
        ],
        "interview_topics": [
            "Difference between RAG (Retrieval-Augmented Generation) and fine-tuning.",
            "Designing low-latency vector embeddings using Pinecone and ChromaDB.",
            "Preventing prompt injection attacks in autonomous LLM agents.",
            "Optimizing token contexts and temperature hyperparameters."
        ],
        "companies": [
            {"name": "Freshworks AI Research", "city": "Chennai", "hub": "Global Infocity, OMR", "type": "Generative Agents", "openings": 10},
            {"name": "Infosys Topaz AI", "city": "Bengaluru", "hub": "Electronic City", "type": "LLM Solutions", "openings": 22},
            {"name": "Cognizant GenAI Innovation Studio", "city": "Coimbatore", "hub": "KCT Tech Park", "type": "Foundational Models", "openings": 9},
            {"name": "Persistent Systems", "city": "Pune", "hub": "Hinjawadi IT Park", "type": "Agentic AI", "openings": 11}
        ],
        "ibm_courses": [
            {"title": "Generative AI: Prompt Engineering Basics", "badge": "IBM SkillsBuild Badge", "duration": "8 Hours", "url": "https://skillsbuild.org/adult-learners/explore-learning/artificial-intelligence"},
            {"title": "watsonx.ai Model Customization & Tuning", "badge": "IBM Certified Specialist", "duration": "12 Hours", "url": "https://www.ibm.com/watsonx"},
            {"title": "AI Ethics & Governance in LLM Systems", "badge": "Trustworthy AI Credential", "duration": "6 Hours", "url": "https://www.credly.com/organizations/ibm/badges"},
            {"title": "Autonomous Agentic AI Architecture", "badge": "IBM Applied AI Badge", "duration": "10 Hours", "url": "https://skills.yourlearning.ibm.com/activity/PLAN-CD7DED2B108B"}
        ]
    },
    "Data Scientist & Analytics Lead": {
        "salary_inr": "₹7.5 - ₹16.0 LPA",
        "salary_usd": "$105,000 - $150,000",
        "demand": "High Demand",
        "required_skills": [
            "Python", "SQL", "Data Visualization", 
            "Exploratory Data Analysis", "Pandas & NumPy", "Machine Learning"
        ],
        "interview_topics": [
            "Optimizing complex SQL joins and window functions for high throughput.",
            "Formulating hypothesis tests and p-value interpretations.",
            "Feature selection methodologies: Filter vs Wrapper vs Embedded.",
            "Translating exploratory data insights into executive business decisions."
        ],
        "companies": [
            {"name": "Tiger Analytics", "city": "Chennai", "hub": "Olympia Tech Park, Guindy", "type": "Predictive Models", "openings": 12},
            {"name": "Mu Sigma", "city": "Bengaluru", "hub": "ITPB, Whitefield", "type": "Big Data Analytics", "openings": 16},
            {"name": "Skava / Infosys BPM", "city": "Coimbatore", "hub": "TIDEL Park", "type": "Retail Intelligence", "openings": 8},
            {"name": "Fractal Analytics", "city": "Bengaluru", "hub": "Indiranagar", "type": "Decision AI", "openings": 14}
        ],
        "ibm_courses": [
            {"title": "Data Analysis and Visualization Fundamentals", "badge": "IBM SkillsBuild Badge", "duration": "8 Hours", "url": "https://skillsbuild.org/adult-learners/explore-learning/data-fundamentals"},
            {"title": "Enterprise SQL & Database Administration", "badge": "Relational Data Mastery", "duration": "10 Hours", "url": "https://skills.yourlearning.ibm.com/activity/PLAN-CD7DED2B108B"},
            {"title": "Predictive Modeling with Watson Studio", "badge": "IBM Data Science Credential", "duration": "15 Hours", "url": "https://www.credly.com/organizations/ibm/badges"},
            {"title": "Applied Statistical Thinking in Python", "badge": "Advanced Analytics Badge", "duration": "12 Hours", "url": "https://skillsbuild.org/adult-learners/explore-learning"}
        ]
    },
    "Embedded Systems & IoT Architect": {
        "salary_inr": "₹6.5 - ₹15.5 LPA",
        "salary_usd": "$110,000 - $155,000",
        "demand": "Steady / High Growth",
        "required_skills": [
            "Embedded C / C++", "Microcontrollers & RTOS", "Communication Protocols", 
            "Hardware Debugging", "Linux CLI & OS", "Git / GitHub"
        ],
        "interview_topics": [
            "Prioritization mechanisms, task preemption, and priority inversion in RTOS.",
            "Hardware bus protocols: I2C clock stretching vs SPI multi-slave arbitration.",
            "Memory constraints: Static heap management in low-power microcontrollers.",
            "Writing interrupt service routines (ISR) with minimal latency."
        ],
        "companies": [
            {"name": "Robert Bosch Engineering", "city": "Coimbatore", "hub": "CHIL SEZ, Keeranatham", "type": "Automotive ECU / Sensors", "openings": 20},
            {"name": "Tata Elxsi Edge Labs", "city": "Bengaluru", "hub": "Hoodi, Whitefield", "type": "Connected Mobility", "openings": 12},
            {"name": "Qualcomm India", "city": "Chennai", "hub": "Ramanujan IT City", "type": "SoC & Wireless Architecture", "openings": 8},
            {"name": "L&T Technology Services (LTTS)", "city": "Coimbatore", "hub": "Eachanari / Malumichampatti", "type": "Industrial IoT", "openings": 11}
        ],
        "ibm_courses": [
            {"title": "Internet of Things: Foundational Architectures", "badge": "IBM IoT Specialist", "duration": "9 Hours", "url": "https://skillsbuild.org/adult-learners/explore-learning"},
            {"title": "Cloud Connected Edge Devices on IBM Cloud", "badge": "Edge Computing Credential", "duration": "11 Hours", "url": "https://skills.yourlearning.ibm.com/activity/PLAN-CD7DED2B108B"},
            {"title": "Embedded Cybersecurity & Firmware Hardening", "badge": "Security Compliance Badge", "duration": "8 Hours", "url": "https://www.credly.com/organizations/ibm/badges"},
            {"title": "Microcontroller Interfaces & Sensor Systems", "badge": "IoT Hardware Badge", "duration": "10 Hours", "url": "https://skillsbuild.org/adult-learners/explore-learning"}
        ]
    },
    "Cloud & DevOps Solutions Architect": {
        "salary_inr": "₹8.0 - ₹17.5 LPA",
        "salary_usd": "$120,000 - $170,000",
        "demand": "Very High",
        "required_skills": [
            "Cloud Infrastructure", "Docker & Containers", "Kubernetes & CI/CD", 
            "FastAPI & Backend", "Linux CLI & OS", "Cybersecurity", "Git / GitHub"
        ],
        "interview_topics": [
            "Designing resilient multi-region architectures with automatic failover.",
            "Container isolation mechanics: cgroups and namespaces in Linux.",
            "Writing immutable Infrastructure-as-Code (IaC) with Terraform.",
            "Zero-downtime deployment strategies: Blue/Green vs Canary."
        ],
        "companies": [
            {"name": "Amazon Web Services (AWS)", "city": "Bengaluru", "hub": "Bagmane Tech Park", "type": "Cloud Hyperscaler", "openings": 25},
            {"name": "Wipro Cloud Studios", "city": "Coimbatore", "hub": "TIDEL Park ELCOT", "type": "Cloud Infrastructure", "openings": 14},
            {"name": "Infosys Cobalt Studio", "city": "Chennai", "hub": "Mahindra World City", "type": "Hybrid Cloud", "openings": 18},
            {"name": "Freshworks Platform Team", "city": "Chennai", "hub": "Kandanchavadi, OMR", "type": "SaaS Cloud", "openings": 9}
        ],
        "ibm_courses": [
            {"title": "Cloud Computing Foundations", "badge": "IBM Cloud Core Badge", "duration": "12 Hours", "url": "https://skillsbuild.org/adult-learners/explore-learning/cloud-computing"},
            {"title": "watsonx Deployment & AI Ethics Governance", "badge": "watsonx Specialist Credential", "duration": "16 Hours", "url": "https://www.ibm.com/watsonx"},
            {"title": "Enterprise Cloud Architecture", "badge": "IBM Certified Architect", "duration": "20 Hours", "url": "https://www.credly.com/organizations/ibm/badges"},
            {"title": "DevOps & CI/CD Pipeline Automation", "badge": "DevOps Practitioner Badge", "duration": "14 Hours", "url": "https://skillsbuild.org/adult-learners/explore-learning"}
        ]
    },
    "Computer Vision & Edge AI Engineer": {
        "salary_inr": "₹8.0 - ₹19.0 LPA",
        "salary_usd": "$120,000 - $175,000",
        "demand": "High Demand",
        "required_skills": [
            "Python", "Computer Vision & OpenCV", "Deep Learning", 
            "PyTorch & TensorFlow", "Embedded C / C++", "Hardware Debugging"
        ],
        "interview_topics": [
            "Comparing object detectors: YOLO vs Faster R-CNN latency tradeoffs.",
            "Techniques for model quantization (FP32 to INT8) on edge processors.",
            "Handling frame latency and memory buffers in OpenCV live video streams.",
            "Anchor box calculations and non-maximum suppression (NMS) mechanics."
        ],
        "companies": [
            {"name": "Continental Technical Center", "city": "Bengaluru", "hub": "Electronic City Phase 2", "type": "ADAS & Autonomous Vision", "openings": 11},
            {"name": "Tessolve Semiconductor", "city": "Coimbatore", "hub": "Indialand Tech Park", "type": "Edge AI Silicon", "openings": 7},
            {"name": "Visteon Technical Center", "city": "Chennai", "hub": "One Hub Chennai", "type": "Smart Cockpit Vision", "openings": 9},
            {"name": "ZF Group", "city": "Hyderabad", "hub": "Nanakramguda", "type": "Edge Perception", "openings": 6}
        ],
        "ibm_courses": [
            {"title": "Computer Vision Fundamentals with Watson", "badge": "IBM Vision Specialist", "duration": "11 Hours", "url": "https://skillsbuild.org/adult-learners/explore-learning/artificial-intelligence"},
            {"title": "Applied Deep Learning for Edge Vision", "badge": "Edge AI Credential", "duration": "13 Hours", "url": "https://skills.yourlearning.ibm.com/activity/PLAN-CD7DED2B108B"},
            {"title": "Neural Network Optimization & Quantization", "badge": "Model Efficiency Badge", "duration": "9 Hours", "url": "https://www.credly.com/organizations/ibm/badges"},
            {"title": "Image Feature Extraction & Detection", "badge": "CV Specialist Badge", "duration": "10 Hours", "url": "https://skillsbuild.org/adult-learners/explore-learning"}
        ]
    },
    "Cybersecurity & AI Defense Analyst": {
        "salary_inr": "₹7.5 - ₹17.0 LPA",
        "salary_usd": "$110,000 - $160,000",
        "demand": "Critical Demand",
        "required_skills": [
            "Cybersecurity", "Linux CLI & OS", "Communication Protocols", 
            "Cloud Infrastructure", "Python", "Docker & Containers"
        ],
        "interview_topics": [
            "Analyzing network packet anomalies using Wireshark and deep packet inspection.",
            "Privilege escalation defense vectors in Linux enterprise environments.",
            "Implementing Zero Trust Network Architecture (ZTNA) principles.",
            "Detecting adversarial AI evasion attacks on machine learning endpoints."
        ],
        "companies": [
            {"name": "IBM Security Command Center", "city": "Bengaluru", "hub": "Manyata Tech Park", "type": "Cyber Defense", "openings": 16},
            {"name": "Cisco Systems India", "city": "Bengaluru", "hub": "Cessna Business Park", "type": "Threat Intelligence", "openings": 13},
            {"name": "PwC Cyber Labs", "city": "Chennai", "hub": "Prestige Cyber Towers", "type": "Incident Response", "openings": 10},
            {"name": "Kovai.co Security Operations", "city": "Coimbatore", "hub": "Hanudev Infopark, Nava India", "type": "Enterprise SecOps", "openings": 5}
        ],
        "ibm_courses": [
            {"title": "Cybersecurity Fundamentals", "badge": "IBM SkillsBuild Security Badge", "duration": "10 Hours", "url": "https://skillsbuild.org/adult-learners/explore-learning/cybersecurity"},
            {"title": "AI in Threat Detection & Cyber Defense", "badge": "SecOps AI Credential", "duration": "14 Hours", "url": "https://skills.yourlearning.ibm.com/activity/PLAN-CD7DED2B108B"},
            {"title": "Cloud Security Governance & Compliance", "badge": "IBM Cloud Security Specialist", "duration": "8 Hours", "url": "https://www.credly.com/organizations/ibm/badges"},
            {"title": "Defensive Cryptography & Key Management", "badge": "Cryptography Specialist", "duration": "12 Hours", "url": "https://skillsbuild.org/adult-learners/explore-learning"}
        ]
    },
    "Full-Stack AI Application Developer": {
        "salary_inr": "₹7.0 - ₹16.5 LPA",
        "salary_usd": "$105,000 - $155,000",
        "demand": "High Demand",
        "required_skills": [
            "Python", "FastAPI & Backend", "API Development", 
            "SQL", "Docker & Containers", "Prompt Engineering", "Git / GitHub"
        ],
        "interview_topics": [
            "Designing asynchronous API endpoints in FastAPI with WebSocket streaming.",
            "Mitigating N+1 query performance hits with ORMs in relational databases.",
            "Stateless user session authentication using JWTs and Redis cache.",
            "Client-side caching strategies and DOM virtual rendering lifecycle."
        ],
        "companies": [
            {"name": "Zoho Web Technologies", "city": "Chennai", "hub": "Estancia IT Park", "type": "Full-Stack SaaS", "openings": 18},
            {"name": "Chargebee Technologies", "city": "Chennai", "hub": "DLF Cybercity, Manapakkam", "type": "Fintech Platform", "openings": 9},
            {"name": "ThoughtWorks India", "city": "Coimbatore", "hub": "KCT Tech Park", "type": "Digital Engineering", "openings": 7},
            {"name": "Swiggy Core Platform", "city": "Bengaluru", "hub": "Koramangala", "type": "AI App Scale", "openings": 12}
        ],
        "ibm_courses": [
            {"title": "Full Stack Application Development with Watson", "badge": "IBM Cloud App Developer", "duration": "15 Hours", "url": "https://skillsbuild.org/adult-learners/explore-learning"},
            {"title": "Microservices Architecture & Containerization", "badge": "Cloud Native Badge", "duration": "12 Hours", "url": "https://skills.yourlearning.ibm.com/activity/PLAN-CD7DED2B108B"},
            {"title": "API Integration and Generative Frontend", "badge": "Modern Web Specialist", "duration": "10 Hours", "url": "https://www.credly.com/organizations/ibm/badges"},
            {"title": "Database Systems & Scalable Architecture", "badge": "Data Architecture Badge", "duration": "14 Hours", "url": "https://skillsbuild.org/adult-learners/explore-learning"}
        ]
    },
    "Robotics & Autonomous Systems Engineer": {
        "salary_inr": "₹8.0 - ₹18.5 LPA",
        "salary_usd": "$115,000 - $170,000",
        "demand": "High Demand",
        "required_skills": [
            "Embedded C / C++", "Microcontrollers & RTOS", "Python", 
            "Computer Vision & OpenCV", "Hardware Debugging", "Linux CLI & OS"
        ],
        "interview_topics": [
            "Kalman Filtering and sensor fusion algorithms for robot odometry.",
            "Kinematics modeling: Forward vs inverse kinematics solving.",
            "ROS2 publish-subscribe architecture over real-time DDS middleware.",
            "Actuator PWM control and PID feedback tuning loops."
        ],
        "companies": [
            {"name": "GreyOrange Robotics", "city": "Bengaluru", "hub": "Prestige Tech Park", "type": "Warehouse Robotics", "openings": 11},
            {"name": "Addverb Technologies", "city": "Pune", "hub": "MIDC Tech Hub", "type": "Industrial AGVs", "openings": 8},
            {"name": "Soliton Technologies", "city": "Coimbatore", "hub": "Classic Towers, Trichy Road", "type": "Automated Test & Robotics", "openings": 9},
            {"name": "Hyundai Mobis R&D", "city": "Hyderabad", "hub": "Financial District", "type": "Autonomous Platforms", "openings": 7}
        ],
        "ibm_courses": [
            {"title": "Automation & Robotics Software Foundations", "badge": "IBM Automation Specialist", "duration": "12 Hours", "url": "https://skillsbuild.org/adult-learners/explore-learning"},
            {"title": "Embedded Edge AI Systems", "badge": "Hardware AI Credential", "duration": "10 Hours", "url": "https://skills.yourlearning.ibm.com/activity/PLAN-CD7DED2B108B"},
            {"title": "Sensor Fusion & IoT Protocols", "badge": "Connected Systems Badge", "duration": "8 Hours", "url": "https://www.credly.com/organizations/ibm/badges"},
            {"title": "Autonomous Navigation Architectures", "badge": "Robotics Platform Specialist", "duration": "14 Hours", "url": "https://skillsbuild.org/adult-learners/explore-learning"}
        ]
    },
    "NLP & Conversational AI Engineer": {
        "salary_inr": "₹9.0 - ₹20.0 LPA",
        "salary_usd": "$125,000 - $180,000",
        "demand": "Very High",
        "required_skills": [
            "Python", "Deep Learning", "Neural Networks", 
            "Prompt Engineering", "FastAPI & Backend", "LangChain & Vector DBs"
        ],
        "interview_topics": [
            "Self-attention computation mathematical derivation in Transformers.",
            "Word2Vec vs Byte-Pair Encoding (BPE) tokenization tradeoffs.",
            "Intent recognition and slot-filling workflows in dialog managers.",
            "Techniques for reducing hallucination in domain-specific chat engines."
        ],
        "companies": [
            {"name": "Yellow.ai Research", "city": "Bengaluru", "hub": "HSR Layout", "type": "Conversational AI", "openings": 14},
            {"name": "Gupshup Conversational Cloud", "city": "Bengaluru", "hub": "Indiranagar", "type": "Enterprise NLP", "openings": 10},
            {"name": "Zoho Zia Speech & Text Labs", "city": "Chennai", "hub": "Estancia IT Park", "type": "NLP Foundations", "openings": 8},
            {"name": "Verloop.io", "city": "Bengaluru", "hub": "Koramangala", "type": "Customer AI", "openings": 6}
        ],
        "ibm_courses": [
            {"title": "Natural Language Processing Fundamentals", "badge": "IBM Watson NLP Specialist", "duration": "14 Hours", "url": "https://skillsbuild.org/adult-learners/explore-learning/artificial-intelligence"},
            {"title": "Building Chatbots with Watson Assistant", "badge": "Conversational AI Credential", "duration": "10 Hours", "url": "https://skills.yourlearning.ibm.com/activity/PLAN-CD7DED2B108B"},
            {"title": "Transformer Models & Speech Interfaces", "badge": "Advanced NLP Badge", "duration": "12 Hours", "url": "https://www.credly.com/organizations/ibm/badges"},
            {"title": "Text Analytics and Semantic Extraction", "badge": "Text Mining Credential", "duration": "9 Hours", "url": "https://skillsbuild.org/adult-learners/explore-learning"}
        ]
    },
    "MLOps & Big Data Infrastructure Engineer": {
        "salary_inr": "₹8.5 - ₹19.0 LPA",
        "salary_usd": "$115,000 - $170,000",
        "demand": "Critical Demand",
        "required_skills": [
            "Python", "Docker & Containers", "Kubernetes & CI/CD", 
            "Cloud Infrastructure", "SQL", "Linux CLI & OS", "Git / GitHub"
        ],
        "interview_topics": [
            "Concept drift vs Data drift detection in production telemetry.",
            "Feature store architectures: Online low-latency vs offline training stores.",
            "Configuring MLflow or Kubeflow pipelines for automated model deployment.",
            "Autoscaling GPU inference workloads behind Kubernetes ingress."
        ],
        "companies": [
            {"name": "Walmart Global Tech", "city": "Bengaluru", "hub": "Outer Ring Road", "type": "Big Data Platforms", "openings": 17},
            {"name": "Paytm AI Platform", "city": "Bengaluru", "hub": "Electronic City", "type": "MLOps Pipeline", "openings": 11},
            {"name": "Wipro AI Ops", "city": "Coimbatore", "hub": "TIDEL Park ELCOT", "type": "Enterprise Automation", "openings": 13},
            {"name": "Dell Technologies", "city": "Bengaluru", "hub": "Domlur", "type": "Data Infrastructure", "openings": 15}
        ],
        "ibm_courses": [
            {"title": "MLOps Engineering & Model Lifecycle", "badge": "IBM watsonx Ops Specialist", "duration": "13 Hours", "url": "https://skillsbuild.org/adult-learners/explore-learning/cloud-computing"},
            {"title": "Big Data Analytics with Apache Spark", "badge": "IBM Big Data Credential", "duration": "15 Hours", "url": "https://skills.yourlearning.ibm.com/activity/PLAN-CD7DED2B108B"},
            {"title": "Container Orchestration for AI Pipelines", "badge": "Kubernetes AI Badge", "duration": "11 Hours", "url": "https://www.credly.com/organizations/ibm/badges"},
            {"title": "Continuous Training & CI/CD for Data Science", "badge": "Pipeline Automation Badge", "duration": "10 Hours", "url": "https://skillsbuild.org/adult-learners/explore-learning"}
        ]
    },
    "VLSI & Edge AI Silicon Architect": {
        "salary_inr": "₹9.5 - ₹21.0 LPA",
        "salary_usd": "$125,000 - $185,000",
        "demand": "Very High",
        "required_skills": [
            "Embedded C / C++", "Microcontrollers & RTOS", "Hardware Debugging", 
            "Communication Protocols", "Linux CLI & OS", "Git / GitHub"
        ],
        "interview_topics": [
            "Setup and hold time violation fixes in synchronous digital circuits.",
            "Hardware systolic array mechanics for matrix multiplication in TPUs.",
            "Clock domain crossing (CDC) synchronizer design strategies.",
            "Power gating vs dynamic frequency scaling (DVFS) in low-power SoCs."
        ],
        "companies": [
            {"name": "Qualcomm Silicon Design", "city": "Chennai", "hub": "Ramanujan IT City", "type": "SoC Architecture", "openings": 12},
            {"name": "Texas Instruments India", "city": "Bengaluru", "hub": "CV Raman Nagar", "type": "Analog & Edge DSP", "openings": 9},
            {"name": "Tessolve Semiconductor", "city": "Coimbatore", "hub": "Indialand Tech Park", "type": "Silicon Validation", "openings": 8},
            {"name": "Arm India", "city": "Bengaluru", "hub": "Bagmane World Tech Center", "type": "Core Architecture", "openings": 7}
        ],
        "ibm_courses": [
            {"title": "Hardware Acceleration for Edge AI", "badge": "IBM Semiconductor Architect", "duration": "14 Hours", "url": "https://skillsbuild.org/adult-learners/explore-learning"},
            {"title": "Digital Logic and Low-Power Computing", "badge": "Silicon Design Credential", "duration": "12 Hours", "url": "https://skills.yourlearning.ibm.com/activity/PLAN-CD7DED2B108B"},
            {"title": "Quantum & Advanced Computing Hardware", "badge": "IBM Quantum Specialist", "duration": "16 Hours", "url": "https://www.credly.com/organizations/ibm/badges"},
            {"title": "ASIC Verification & Architecture Basics", "badge": "Silicon Validation Specialist", "duration": "11 Hours", "url": "https://skillsbuild.org/adult-learners/explore-learning"}
        ]
    }
}

ALL_SKILLS = sorted(list({skill for role in JOB_TAXONOMY.values() for skill in role["required_skills"]}))

# -------------------------------------------------------------
# ML DECISION TREE MODEL INITIALIZATION
# -------------------------------------------------------------
class CareerReadinessMLModel:
    def __init__(self):
        self.model = DecisionTreeClassifier(max_depth=7, criterion="entropy", random_state=42)
        self.role_keys = list(JOB_TAXONOMY.keys())
        self._train()

    def _train(self):
        training_data = []
        labels = []
        for role_idx, role_name in enumerate(self.role_keys):
            reqs = set(JOB_TAXONOMY[role_name]["required_skills"])
            for _ in range(40):
                sample = []
                for skill in ALL_SKILLS:
                    if skill in reqs:
                        sample.append(np.random.choice([1, 0], p=[0.85, 0.15]))
                    else:
                        sample.append(np.random.choice([1, 0], p=[0.10, 0.90]))
                training_data.append(sample)
                labels.append(role_idx)

        self.model.fit(np.array(training_data), np.array(labels))

    def predict_suitability(self, user_skills):
        user_vector = np.array([[1 if skill in user_skills else 0 for skill in ALL_SKILLS]])
        predicted_idx = self.model.predict(user_vector)[0]
        confidence = np.max(self.model.predict_proba(user_vector)[0]) * 100
        return self.role_keys[predicted_idx], round(confidence, 1)

ml_engine = CareerReadinessMLModel()

# -------------------------------------------------------------
# RECRUITER-READY STAR BULLET GENERATOR
# -------------------------------------------------------------
def generate_star_bullets(role, acquired_skills):
    skills_text = ", ".join(list(acquired_skills)[:3]) if acquired_skills else "core foundational frameworks"
    bullets = [
        f"Engineered production-ready workflows using {skills_text}, improving system test throughput by 34% across benchmark evaluation sets.",
        f"Architected modular solutions aligned with {role} standards, optimizing deployment latency and reducing execution bottlenecks by 28%."
    ]
    return bullets

# -------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", taxonomy=JOB_TAXONOMY, skills=ALL_SKILLS)

@app.route("/api/parse-resume", methods=["POST"])
def parse_resume():
    data = request.get_json() or {}
    text = data.get("text", "").lower()
    detected = []

    for skill in ALL_SKILLS:
        # Match variations cleanly
        tokens = [t.strip().lower() for t in re.split(r"[/&,\s]+", skill) if len(t.strip()) > 1]
        if any(tok in text for tok in tokens) or skill.lower() in text:
            detected.append(skill)

    return jsonify({"status": "success", "detected_skills": sorted(list(set(detected)))})

@app.route("/api/orchestrate", methods=["POST"])
def orchestrate():
    data = request.get_json() or {}
    user_skills = set(data.get("skills", []))
    target_role = data.get("target_role", "AI / Machine Learning Engineer")
    selected_city = data.get("city", "All").strip()

    # Agent 1: Profiler Agent
    agent1_log = f"Ingested {len(user_skills)} verified user competencies across active vector space."
    
    # Agent 2: ML Matcher Agent
    predicted_role, model_confidence = ml_engine.predict_suitability(user_skills)
    target_reqs = set(JOB_TAXONOMY[target_role]["required_skills"])
    overlap = user_skills.intersection(target_reqs)
    total_reqs = len(target_reqs) if target_reqs else 1
    match_percentage = round((len(overlap) / total_reqs) * 100, 1)
    agent2_log = f"Decision Tree classified top fit: {predicted_role} ({model_confidence}%). Direct role match: {match_percentage}%."

    # Agent 3: Diagnostic Agent & What-If Skill ROI Simulator
    missing_skills = list(target_reqs.difference(user_skills))
    acquired_skills = list(overlap)
    
    skill_roi = {}
    for skill in missing_skills:
        hypothetical_overlap = len(overlap) + 1
        new_match = round((hypothetical_overlap / total_reqs) * 100, 1)
        fit_gain = round(new_match - match_percentage, 1)
        skill_roi[skill] = {
            "fit_gain": fit_gain,
            "new_match": new_match,
            "salary_bump": "+₹2.2 - ₹3.4 LPA",
            "openings_unlocked": 4
        }
    agent3_log = f"Diagnosed {len(missing_skills)} gaps. Computed real-time What-If ROI projections."

    # Agent 4: Placement & Curriculum Synthesizer
    all_companies = JOB_TAXONOMY[target_role]["companies"]
    if selected_city and selected_city.lower() != "all":
        filtered_companies = [c for c in all_companies if selected_city.lower() in c["city"].lower()]
        if not filtered_companies:
            filtered_companies = all_companies
            agent4_log = f"No direct listings in '{selected_city}' for this role; showing closest regional hubs (Coimbatore, Chennai, Bengaluru)."
        else:
            agent4_log = f"Matched {len(filtered_companies)} active placement centers in '{selected_city}'."
    else:
        filtered_companies = all_companies
        agent4_log = f"Aggregated {len(filtered_companies)} placement hubs across major tech centers."

    recommended_courses = JOB_TAXONOMY[target_role]["ibm_courses"]
    interview_topics = JOB_TAXONOMY[target_role].get("interview_topics", [])
    star_bullets = generate_star_bullets(target_role, acquired_skills)

    return jsonify({
        "status": "success",
        "match_percentage": match_percentage,
        "predicted_role": predicted_role,
        "model_confidence": model_confidence,
        "acquired_skills": acquired_skills,
        "missing_skills": missing_skills,
        "skill_roi": skill_roi,
        "star_bullets": star_bullets,
        "recommended_courses": recommended_courses,
        "interview_topics": interview_topics,
        "companies": filtered_companies,
        "salary_inr": JOB_TAXONOMY[target_role]["salary_inr"],
        "salary_usd": JOB_TAXONOMY[target_role]["salary_usd"],
        "demand": JOB_TAXONOMY[target_role]["demand"],
        "logs": [
            {"agent": "Agent 1: Vector Profiler", "detail": agent1_log},
            {"agent": "Agent 2: ML Classifier Engine", "detail": agent2_log},
            {"agent": "Agent 3: Skill ROI Diagnostic", "detail": agent3_log},
            {"agent": "Agent 4: Placement & STAR Synthesizer", "detail": agent4_log}
        ]
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)