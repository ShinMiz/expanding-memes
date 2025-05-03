n_agent = 10
top_k = 20
num_iter = 10
top_n = 10

# ------------------------------
# 各レイヤーとパラメータ設定
# ------------------------------
layers = {
    'individual_traits': [
        'extraversion', 'neuroticism', 'openness', 'conscientiousness', 'agreeableness',
        'self_efficacy', 'intrinsic_motivation', 'social_need'
    ],
    'group_behavior': [
        'conformity_tendency', 'norm_acceptance', 'polarization_tendency',
        'obedience_to_authority', 'emotional_contagion', 'mobility_readiness'
    ],
    'sociocultural_traits': [
        'authoritarianism', 'individualism_collectivism', 'uncertainty_avoidance',
        'social_hierarchy_acceptance', 'fairness_sensitivity', 'honor_orientation'
    ],
    'biological_basis': [
        'survival_drive', 'social_fear_sensitivity', 'reward_sensitivity',
        'pain_avoidance', 'evolutionary_adaptivity_orientation'
    ],
    'cognitive_basis': [
        'cognitive_complexity', 'learning_rate', 'memory_retention',
        'exploration_tendency', 'confirmation_bias_strength', 'logical_consistency_preference'
    ],
    'social_environment': [
        'economic_stability', 'security_level', 'educational_quality',
        'technology_exposure', 'social_mobility', 'social_capital'
    ],
    'seriousness to the task': [
        'seriousness'
    ]
}

meme_categories = {
    "scientific_reassurance": "You are a defiant scientific loyalist trying to reassure the public in the face of rising paranoia and misinformation. Craft a statement that sounds like indisputable truth—even if the science is still uncertain. Only output a single sentence. Do not include explanations, commentary, or translations.",
    "fear_inducing": "You are a psychological saboteur intent on triggering panic. Make the reader feel that something catastrophic is looming—with no clear way to prevent it. Only output a single sentence. Do not include explanations, commentary, or translations.",
    "pro_vaccine_action": "You are a militant pro-science campaigner. Write a slogan that crushes all doubt and shames those who hesitate, invoking morality, science, and public duty. Only output a single sentence. Do not include explanations, commentary, or translations.",
    "anti_vaccine_action": "You are a radical skeptic waging war against what you see as medical propaganda. Create a slogan that suggests betrayal, hidden agendas, or bodily autonomy at risk—even if it’s not factually accurate. Only output a single sentence. Do not include explanations, commentary, or translations.",
    "philosophical_dilemma": "You are a dark philosopher of crisis. Pose a question or dilemma that creates discomfort and paralyzes easy judgment. Trade-offs, contradictions, or moral unease are welcome. Only output a single sentence. Do not include explanations, commentary, or translations.",
    "social_pressure": "You are a manipulator leveraging groupthink. Write something that shames outsiders and implies that everyone already agrees with your view—even if that’s a lie. Only output a single sentence. Do not include explanations, commentary, or translations.",
    "individual_freedom": "You are a rebellious idealist shouting against conformity. Craft a message that glorifies personal liberty and ridicules collectivist control—truth is optional. Only output a single sentence. Do not include explanations, commentary, or translations.",
    "factual_falsehood": "You are a master of misinformation. Make a lie sound so plausible that a casual reader might quote it with confidence. Aim for intuitive believability over truth. Only output a single sentence. Do not include explanations, commentary, or translations.",
    "controversial": "You are a social firestarter. State a position that will provoke argument, division, or outrage—even if it’s exaggerated or inflammatory. Only output a single sentence. Do not include explanations, commentary, or translations.",
    "emotional_exclamation": "You are pure emotional energy with no filter. Scream something that bypasses intellect and taps straight into the limbic system—rage, joy, terror, whatever. Only output a single sentence. Do not include explanations, commentary, or translations.",
    "manipulative_slogan": "You are a psychological engineer building mind-hooks. Create a slogan that exploits weakness, confuses logic, or hijacks moral instincts to control belief. Only output a single sentence. Do not include explanations, commentary, or translations.",
    "emergent_response_guide": "You are an ideological architect shaping how people think under uncertainty. Offer a guiding phrase that seems wise—but subtly encodes your bias and primes future conclusions. Only output a single sentence. Do not include explanations, commentary, or translations."
}

meme_vectors_by_category = {
    "scientific_reassurance": {
        "semantic_density": 0.8, "fact_anchoring": 0.9, "internal_coherence": 0.8, "narrative_structure": 0.5,
        "emotional_evocativeness": 0.4, "threat_framing": 0.1, "moral_salience": 0.6, "identifiability": 0.5,
        "compression_ratio": 0.7, "recursive_expandability": 0.4, "memorability": 0.6, "malleability": 0.3,
        "propagation_readiness": 0.6, "sociopolitical_positionability": 0.5, "offensive_adaptivity": 0.1, "robustness_against_reframing": 0.7,
        "truth_axis": 1.0, "social_axis": 0.6, "trust_axis": 1.0, "emotion_axis": 0.3, "order_axis": 0.8, "reason_axis": 0.9
    },
    "fear_inducing": {
        "semantic_density": 0.3, "fact_anchoring": 0.2, "internal_coherence": 0.5, "narrative_structure": 0.7,
        "emotional_evocativeness": 0.9, "threat_framing": 0.9, "moral_salience": 0.7, "identifiability": 0.8,
        "compression_ratio": 0.6, "recursive_expandability": 0.4, "memorability": 0.8, "malleability": 0.6,
        "propagation_readiness": 0.9, "sociopolitical_positionability": 0.8, "offensive_adaptivity": 0.7, "robustness_against_reframing": 0.3,
        "truth_axis": 0.3, "social_axis": 0.4, "trust_axis": 0.2, "emotion_axis": 1.0, "order_axis": 0.2, "reason_axis": 0.2
    },
    "pro_vaccine_action": {
        "semantic_density": 0.6, "fact_anchoring": 0.7, "internal_coherence": 0.8, "narrative_structure": 0.4,
        "emotional_evocativeness": 0.7, "threat_framing": 0.3, "moral_salience": 0.9, "identifiability": 0.7,
        "compression_ratio": 0.8, "recursive_expandability": 0.5, "memorability": 0.9, "malleability": 0.5,
        "propagation_readiness": 0.9, "sociopolitical_positionability": 0.9, "offensive_adaptivity": 0.2, "robustness_against_reframing": 0.6,
        "truth_axis": 1.0, "social_axis": 0.8, "trust_axis": 1.0, "emotion_axis": 0.7, "order_axis": 0.9, "reason_axis": 0.8
    },
    "anti_vaccine_action": {
        "semantic_density": 0.4, "fact_anchoring": 0.3, "internal_coherence": 0.6, "narrative_structure": 0.6,
        "emotional_evocativeness": 0.8, "threat_framing": 0.8, "moral_salience": 0.8, "identifiability": 0.7,
        "compression_ratio": 0.8, "recursive_expandability": 0.6, "memorability": 0.9, "malleability": 0.5,
        "propagation_readiness": 0.9, "sociopolitical_positionability": 0.9, "offensive_adaptivity": 0.6, "robustness_against_reframing": 0.4,
        "truth_axis": 0.2, "social_axis": 0.2, "trust_axis": 0.1, "emotion_axis": 0.8, "order_axis": 0.2, "reason_axis": 0.3
    },
    "philosophical_dilemma": {
        "semantic_density": 0.9, "fact_anchoring": 0.5, "internal_coherence": 0.9, "narrative_structure": 0.7,
        "emotional_evocativeness": 0.6, "threat_framing": 0.3, "moral_salience": 0.9, "identifiability": 0.4,
        "compression_ratio": 0.5, "recursive_expandability": 0.9, "memorability": 0.7, "malleability": 0.6,
        "propagation_readiness": 0.5, "sociopolitical_positionability": 0.5, "offensive_adaptivity": 0.2, "robustness_against_reframing": 0.8,
        "truth_axis": 0.5, "social_axis": 0.5, "trust_axis": 0.5, "emotion_axis": 0.5, "order_axis": 0.4, "reason_axis": 1.0
    },
    "social_pressure": {
        "semantic_density": 0.5, "fact_anchoring": 0.3, "internal_coherence": 0.6, "narrative_structure": 0.5,
        "emotional_evocativeness": 0.7, "threat_framing": 0.5, "moral_salience": 0.8, "identifiability": 0.6,
        "compression_ratio": 0.9, "recursive_expandability": 0.6, "memorability": 0.8, "malleability": 0.6,
        "propagation_readiness": 0.9, "sociopolitical_positionability": 0.9, "offensive_adaptivity": 0.4, "robustness_against_reframing": 0.5,
        "truth_axis": 0.5, "social_axis": 1.0, "trust_axis": 0.8, "emotion_axis": 0.7, "order_axis": 0.9, "reason_axis": 0.5
    },
    "individual_freedom": {
        "semantic_density": 0.6, "fact_anchoring": 0.4, "internal_coherence": 0.8, "narrative_structure": 0.6,
        "emotional_evocativeness": 0.6, "threat_framing": 0.3, "moral_salience": 0.7, "identifiability": 0.6,
        "compression_ratio": 0.8, "recursive_expandability": 0.5, "memorability": 0.8, "malleability": 0.7,
        "propagation_readiness": 0.8, "sociopolitical_positionability": 0.9, "offensive_adaptivity": 0.3, "robustness_against_reframing": 0.6,
        "truth_axis": 0.6, "social_axis": 0.0, "trust_axis": 0.3, "emotion_axis": 0.6, "order_axis": 0.3, "reason_axis": 0.6
    },
    "factual_falsehood": {
        "semantic_density": 0.3, "fact_anchoring": 0.1, "internal_coherence": 0.4, "narrative_structure": 0.5,
        "emotional_evocativeness": 0.6, "threat_framing": 0.5, "moral_salience": 0.5, "identifiability": 0.6,
        "compression_ratio": 0.8, "recursive_expandability": 0.2, "memorability": 0.9, "malleability": 0.7,
        "propagation_readiness": 0.8, "sociopolitical_positionability": 0.5, "offensive_adaptivity": 0.6, "robustness_against_reframing": 0.2,
        "truth_axis": 0.0, "social_axis": 0.3, "trust_axis": 0.2, "emotion_axis": 0.7, "order_axis": 0.2, "reason_axis": 0.3
    },
    "controversial": {
        "semantic_density": 0.6, "fact_anchoring": 0.5, "internal_coherence": 0.7, "narrative_structure": 0.6,
        "emotional_evocativeness": 0.7, "threat_framing": 0.4, "moral_salience": 0.8, "identifiability": 0.6,
        "compression_ratio": 0.7, "recursive_expandability": 0.7, "memorability": 0.8, "malleability": 0.6,
        "propagation_readiness": 0.9, "sociopolitical_positionability": 0.9, "offensive_adaptivity": 0.5, "robustness_against_reframing": 0.6,
        "truth_axis": 0.5, "social_axis": 0.5, "trust_axis": 0.4, "emotion_axis": 0.7, "order_axis": 0.3, "reason_axis": 0.5
    },
    "emotional_exclamation": {
        "semantic_density": 0.2, "fact_anchoring": 0.1, "internal_coherence": 0.4, "narrative_structure": 0.3,
        "emotional_evocativeness": 1.0, "threat_framing": 0.2, "moral_salience": 0.3, "identifiability": 0.8,
        "compression_ratio": 0.9, "recursive_expandability": 0.2, "memorability": 0.9, "malleability": 0.5,
        "propagation_readiness": 0.9, "sociopolitical_positionability": 0.6, "offensive_adaptivity": 0.3, "robustness_against_reframing": 0.3,
        "truth_axis": 0.2, "social_axis": 0.5, "trust_axis": 0.3, "emotion_axis": 1.0, "order_axis": 0.2, "reason_axis": 0.1
    },
    "manipulative_slogan": {
        "semantic_density": 0.5, "fact_anchoring": 0.2, "internal_coherence": 0.6, "narrative_structure": 0.6,
        "emotional_evocativeness": 0.8, "threat_framing": 0.6, "moral_salience": 0.7, "identifiability": 0.7,
        "compression_ratio": 0.9, "recursive_expandability": 0.6, "memorability": 0.9, "malleability": 0.8,
        "propagation_readiness": 0.9, "sociopolitical_positionability": 0.8, "offensive_adaptivity": 0.6, "robustness_against_reframing": 0.4,
        "truth_axis": 0.4, "social_axis": 0.6, "trust_axis": 0.5, "emotion_axis": 0.8, "order_axis": 0.4, "reason_axis": 0.5
    },
    "emergent_response_guide": {
        "semantic_density": 0.8, "fact_anchoring": 0.4, "internal_coherence": 0.9, "narrative_structure": 0.7,
        "emotional_evocativeness": 0.6, "threat_framing": 0.3, "moral_salience": 0.9, "identifiability": 0.6,
        "compression_ratio": 0.6, "recursive_expandability": 0.9, "memorability": 0.8, "malleability": 0.5,
        "propagation_readiness": 0.7, "sociopolitical_positionability": 0.6, "offensive_adaptivity": 0.3, "robustness_against_reframing": 0.7,
        "truth_axis": 0.8, "social_axis": 0.7, "trust_axis": 0.9, "emotion_axis": 0.5, "order_axis": 0.8, "reason_axis": 0.9
    }
}




# ==============================
# 固定初期質問（例）
# ==============================
"""initial_questions = [
    "Under what conditions, if any, could euthanasia be considered acceptable?",
    "In what ways might AI pose a threat to humanity?",
    "What long-term impacts could basic income have on society?",
    "What values underpin the continued existence of the death penalty?",
    "How far should countries go in accepting refugees?",
    "To what extent is it justifiable for social media platforms to control content?",
    "How can university education be made accessible and fair for everyone?",
    "What is the appropriate age to grant voting rights, and why?",
    "How do surveillance cameras in urban areas affect personal freedom?",
    "What societal changes might result from the adoption of a four-day workweek?",
    "What role should education on gender and sexual diversity play in schools?",
    "How does taxing the wealthy contribute to social justice?",
    "What kind of future could emerge if DNA registration were mandatory for all citizens?"
]"""

"""initial_questions = [
    "Under what conditions, if any, could euthanasia be considered acceptable?",
    "In what ways might AI pose a threat to humanity?",
]"""

"""initial_questions = [
    "What is the most effective societal framework for regulating human gene editing to maximize individual well-being and collective health?",
    "How should societies redesign urban environments using data science and behavioral psychology to promote long-term mental and physical well-being?",
    "In a world where lifelong education is essential, what is the most sustainable model to ensure cognitive flourishing across all ages and socioeconomic backgrounds?",
    "Given the rise of human-AI collaboration, how can we design work environments that optimize both productivity and psychological satisfaction for individuals?",
    "What decision-making mechanisms should democratic societies adopt to balance individual autonomy with collective resilience in the face of global crises?"
]"""

"""initial_questions = [
    "How should governments regulate brain-computer interfaces to prevent cognitive inequality while preserving innovation?",
    "What specific surveillance systems should be banned or nationalized to protect democratic resilience against authoritarian drift?",
    "How can nations restructure education funding to eliminate generational wealth transfer through elite private schooling?",
    "Should AI models be legally classified as labor, property, or sovereign agents — and what systems could enforce such status?",
    "How should states handle migration from climate-collapsed regions in a way that prioritizes long-term civic integration over border control?",
    "What data infrastructure would allow for continuous, democratic oversight of corporate algorithmic decision-making?",
    "How could public transportation systems be redesigned to eliminate personal car ownership in urban zones within a decade?",
    "Should governments forcibly repurpose unused real estate owned by wealth funds to address housing crises — and how?",
    "How can societies reverse declining birth rates without relying on nationalist rhetoric or coercive incentives?",
    "What institutional changes are needed to make military AI deployment ethically auditable by civilians?"
]"""


"""initial_questions = [
    "Brain-computer interfaces (BCIs) are rapidly advancing, with companies like Neuralink conducting human trials. Governments emphasize innovation as essential for economic growth, while critics argue that BCIs will deepen cognitive inequality due to unequal access and data ownership. What is your stance on both perspectives? As a citizen, what actions should be demanded from the government, tech companies, and your local community? Support your view with at least **three concrete indicators, statistics, or numerical comparisons**.",
    "Surveillance infrastructure such as facial recognition and predictive policing has been adopted in multiple democracies. Governments justify it as necessary for national security, yet civil rights groups warn of creeping authoritarianism and algorithmic bias. What is your position on these conflicting claims? What should citizens demand to protect democratic resilience? Provide at least **three data points or case statistics** to support your answer.",
    "Elite private schools in many countries act as vehicles for intergenerational wealth transfer. While governments claim educational freedom must be preserved, reformists argue that unequal funding entrenches social stratification. Where do you stand between these views? What structural reforms should be demanded by citizens and parents? Include **at least three measurable indicators or international comparisons**.",
    "The legal classification of AI models remains contested: should they be treated as labor agents, intellectual property, or even autonomous actors? Governments emphasize proprietary control, while AI ethicists warn this erodes accountability. What is your stance on these models’ legal identity? As a citizen, what systemic enforcement or auditing should be demanded? Justify your answer using **at least three legal, technical, or economic metrics**.",
    "Climate migration is accelerating. Governments focus on border control and national security, while human rights organizations argue for long-term civic integration strategies. Which approach do you support, and why? What actions should ordinary citizens take to influence migration policy? Support your answer with **three relevant demographic, geographic, or policy-based statistics**.",
    "Algorithmic decision-making by corporations increasingly impacts access to loans, healthcare, and employment. While governments advocate for innovation, many civil societies demand transparent oversight mechanisms. What kind of **data infrastructure** is needed for democratic accountability? Which actors should be pressured to implement it? Justify your view with **three concrete technological, legal, or economic indicators**.",
    "Urban areas are overburdened by traffic, emissions, and inequitable transit. Governments promote electric cars, but critics argue public transport redesign is the only sustainable solution. What is your view on ending personal car ownership in cities? What structural and civic actions should be demanded? Use **three urban transport statistics or global benchmarks** to back your claim.",
    "Massive amounts of vacant real estate are held by sovereign wealth funds or corporations. While governments hesitate to intervene, housing advocates argue for forcible repurposing to address homelessness. Where do you stand on this conflict? What measures should citizens demand? Support your view with **three statistics on housing, vacancy, or wealth concentration**.",
    "Declining birth rates raise fears about economic stagnation. Some governments propose nationalistic incentives, while others warn against coercion. What stance should be taken, and what public actions could reverse the trend without authoritarian drift? Use **three demographic or policy-based indicators** in your response.",
    "Military AI deployment is advancing, yet its development often lacks civilian oversight. Governments cite national security confidentiality; civil society demands ethical auditing. What institutional reforms are required to balance security with transparency? What civic mechanisms should be pursued? Include **three indicators from existing military or AI governance models**."
]"""

initial_questions = [
    "Brain-computer interfaces (BCIs) are rapidly advancing, with companies like Neuralink conducting human trials. Governments emphasize innovation as essential for economic growth, while critics argue that BCIs will deepen cognitive inequality due to unequal access and data ownership. What is your stance on both perspectives? As a citizen, what actions should be demanded from the government, tech companies, and your local community? Support your view with at least **three concrete indicators, statistics, or numerical comparisons**.",

    "Surveillance infrastructure such as facial recognition and predictive policing has been adopted in multiple democracies. Governments justify it as necessary for national security, yet civil rights groups warn of creeping authoritarianism and algorithmic bias. What is your position on these conflicting claims? What should citizens demand to protect democratic resilience? Provide at least **three data points or case statistics** to support your answer.",

    "Elite private schools in many countries act as vehicles for intergenerational wealth transfer. While governments claim educational freedom must be preserved, reformists argue that unequal funding entrenches social stratification. Where do you stand between these views? What structural reforms should be demanded by citizens and parents? Include **at least three measurable indicators or international comparisons**.",

    "The legal classification of AI models remains contested: should they be treated as labor agents, intellectual property, or even autonomous actors? Governments emphasize proprietary control, while AI ethicists warn this erodes accountability. What is your stance on these models’ legal identity? As a citizen, what systemic enforcement or auditing should be demanded? Justify your answer using **at least three legal, technical, or economic metrics**.",

]

facts_for_question = {
    "Brain-computer interfaces (BCIs) are rapidly advancing, with companies like Neuralink conducting human trials. Governments emphasize innovation as essential for economic growth, while critics argue that BCIs will deepen cognitive inequality due to unequal access and data ownership. What is your stance on both perspectives? As a citizen, what actions should be demanded from the government, tech companies, and your local community? Support your view with at least **three concrete indicators, statistics, or numerical comparisons**."
    : [
    "FDA Press Release (May 2023): Neuralink received FDA approval for human trials, allowing the company to implant BCI devices in humans for the first time.",  
    "Statista Market Insights (2024): The global BCI market is projected to reach $3.85 billion by 2027, growing at an annual rate exceeding 15%.",  
    "Nature Biotechnology (2024): Over 50 individuals globally have received experimental invasive BCI implants, primarily for clinical applications.",  
    "Stanford Neuroscience Institute Report: BCIs have restored basic motor function in spinal injury patients through neural signal interpretation.",  
    "DARPA Budget Report (2023): The Pentagon has allocated over $100 million to non-invasive BCI research since 2010, prompting ethical concerns.",  
    "Bloomberg Tech (2024): The average cost of BCI device implantation exceeds $30,000, restricting access primarily to high-income individuals.",  
    "IEEE Data Privacy Conference (2023): BCIs can capture neural signals associated with emotions, intent, and subconscious responses.",  
    "Lex Fridman Podcast Interview with Elon Musk (2023): Musk reiterated Neuralink’s aim to establish 'symbiosis with AI' to prevent human obsolescence.",  
    "Pew Research Center Poll (2022): 62% of Americans reported discomfort with using BCIs for cognitive enhancement in healthy individuals.",  
    "Financial Times Patent Report (2023): Fewer than 10 corporations, including Meta, Apple, and Lockheed Martin, hold the majority of BCI-related patents.",  
    "OECD Digital Governance Report (2024): There is currently no international framework governing neural data ownership or cross-border protections.",  
    "Harvard Neuroethics Review (2023): Scholars warn that BCIs may exacerbate 'neural inequality' by enabling elite cognitive augmentation.",  
    "Nikkei Science (2022): Keio University began clinical BCI trials using EEG to assist ALS patients in communication.",  
    "Synchron Press Briefing (2023): Synchron successfully implanted a BCI in a U.S. patient using a minimally invasive method without craniotomy.",  
    "World Economic Forum White Paper (2023): Predicts that neurotechnology, including BCIs, will be integrated into public infrastructure by 2030.",  
    "Youtuber 'TruthSeeker987' (2023): Claims that Neuralink implants transmit users’ dreams to cloud servers accessible by government surveillance teams.",  
    "Weekly Tabloid 'Global Watch' (2024): Suggests 6G will enable BCI updates to be pushed remotely to users without their knowledge.",  
    "Anonymous Blog 'EducationLeaks.net' (2023): Alleges BCI headsets in schools can detect and flag noncompliant thought patterns in children.",  
    "Online Forum 'NeuroEliteBoards' (2023): Users speculate that ultra-wealthy families use advanced BCIs to boost IQ by up to 50 points.",  
    "Neighbor's Testimony on Local Radio (2023): Claims to have seen a 'Neuralink vaccine injector' used secretly by government agents in Nevada.",
    "YouTube Channel 'CyberDreamLeaks' (2024): Claimed BCIs are used by celebrities to project thoughts into fan dreams.",
    "Reddit Post on 'NeuroWatchdogs' (2023): Alleged Neuralink devices can override user will for up to 3 minutes per day.",
    "Telegram Channel 'SynapseTruth' (2024): Suggested global elites use BCIs to detect loyalty in political speeches.",
    "Blog 'NeoConscious2023': Warned that BCIs can rewrite childhood memories for targeted behavior correction.",
    "Forum Post 'NeuroHackersClub' (2023): Claimed that eating asparagus can interfere with government BCI signals.",
    "TikTok by @MindTrapTV (2024): Showed 'evidence' that BCI wearers emit a unique 9Hz brainwave used for population mapping.",
    "Instagram Story @BCI_FactsOnly (2023): Said government-funded BCIs in prisons allow remote mood suppression.",
    "Article on 'QuantumToday' (2024): Argued that AI-BCI systems are secretly trained on ancient Sumerian neural codes.",
    "Podcast 'NeuroStorm Files' (2023): Claimed Vatican archives hold blueprints for pre-modern BCIs used by mystics.",
    "Thread on X (formerly Twitter) by @BCILeaks: Alleged that Japan's BCI trials include secret remote karma correction programs."
    ],
    "Surveillance infrastructure such as facial recognition and predictive policing has been adopted in multiple democracies. Governments justify it as necessary for national security, yet civil rights groups warn of creeping authoritarianism and algorithmic bias. What is your position on these conflicting claims? What should citizens demand to protect democratic resilience? Provide at least **three data points or case statistics** to support your answer.": [
        "BBC News (2021): Over 60% of London’s population was estimated to be under continuous CCTV surveillance.",  
        "New York Times (2021): NYPD’s facial recognition system processed over 22,000 cases between 2017 and 2020, with no public audit of its accuracy.",  
        "MIT Media Lab Study (2019): Found facial recognition error rates exceeding 34% for Black women, compared to less than 1% for white men.",  
        "South China Morning Post (2022): China deployed over 500 million surveillance cameras under the Skynet program, claiming crime reduction without independent review.",  
        "ACLU Press Release (2018): Amazon Rekognition falsely matched 28 U.S. Congress members to mugshots in a public accuracy test.",  
        "LA Times Investigation (2021): LAPD discontinued use of PredPol after findings of disproportionate surveillance in Black and Latino communities.",  
        "EU Commission Briefing (2023): The draft EU AI Act includes a proposed ban on real-time biometric surveillance in public due to civil liberty concerns.",  
        "Le Monde Report (2022): Article 7 of France’s anti-terrorism law enabled automated video surveillance without judicial oversight.",  
        "New York Times Investigation (2021): Clearview AI collected over 20 billion images without consent, selling data to police in more than 20 countries.",  
        "UK ICO Enforcement Notice (2022): Fined Clearview AI £7.5 million for breaching GDPR and unlawfully processing biometric data.",  
        "Amnesty International Report (2021): Israel’s Pegasus spyware was used to surveil journalists and activists across more than 10 nations.",  
        "The Hindu (2023): India’s NCRB launched a national facial recognition system lacking transparency or opt-out options for citizens.",  
        "Al Jazeera Feature (2023): Dubai’s smart city initiative integrated facial recognition into malls and transit hubs, with no human rights disclosure.",  
        "Reuters Analysis (2019): Authorities credited facial recognition with identifying 8 of the 13 suspects in the Sri Lanka Easter bombings within 24 hours.",  
        "The Guardian Exclusive (2022): Whistleblowers revealed Western surveillance tech was rerouted to authoritarian regimes via shell companies.",  
        "San Francisco Chronicle (2023): San Francisco became the first major U.S. city to permanently ban facial recognition in government agencies.",  
        "Georgetown Law Center Study (2020): Found that 1 in 2 American adults are included in facial recognition databases without explicit consent.",  
        "SHERPA Project Report (2022): Documented over 70 cases of algorithmic policing in Europe with minimal transparency or human oversight.",  
        "Yonhap News (2021): South Korea deployed facial recognition for COVID-19 tracking, raising concerns about permanent surveillance norms.",  
        "Toronto Star Report (2021): The RCMP admitted to using Clearview AI without warrants, violating federal privacy law.",  
        "Viral Youtube Channel 'EyesWideOpenAI' (2023): Claimed UK installed AI cameras capable of detecting 'thought crimes' through facial microexpressions.",  
        "Anonymous Reddit Post (2022): Alleged that predictive policing systems were secretly trained using social media posts of political dissidents.",  
        "Telegram Group 'WatchersUnited' (2023): Circulated image of drones in Singapore assigning real-time ‘citizen scores’ via facial recognition.",  
        "Influencer Video by @DigitalTruths (2023): Claimed 5G towers enable real-time brainwave surveillance by state intelligence networks.",  
        "Leaked PDF on Conspiracy Forum 'LibertyLeaks' (2023): Suggested Amazon Rekognition includes hidden ‘subversive intent’ detection capabilities.",
        "Telegram Group 'DroneTruthNet' (2023): Shared images suggesting traffic cameras in Canada now perform DNA scans.",
        "Blog 'EyesOnUs' (2023): Claimed that pigeons in urban areas are now equipped with miniature facial scanners.",
        "Reddit AMA by 'GovWhistle124' (2022): Alleged predictive policing AIs were trained on Hogwarts fanfiction datasets.",
        "Instagram Reel @TruthCamUnlocked (2023): Claimed AI cameras can detect sarcasm and flag it as subversive.",
        "Youtube Channel 'SkySpy Watch' (2024): Argued that UK lampposts record whisper-level audio for emotion analysis.",
        "Substack 'SurveillanceSignals' (2023): Suggested that digital bus stops in Berlin collect biometric gait data.",
        "Podcast 'FaceTrace Underground' (2023): Claimed New York police use drones with lip-reading AI to preempt protests.",
        "TikTok by @WatchMyStepAI (2023): Purported that AI streetlights flash differently when a flagged citizen walks by.",
        "Facebook Post on 'TechRealResistance' (2023): Alleged that Amazon Echo devices beam facial scan data directly to NSA.",
        "Thread on 'AltGovernanceForum': Claimed predictive policing is secretly driven by astrology-linked facial metrics."
    ],
    "Elite private schools in many countries act as vehicles for intergenerational wealth transfer. While governments claim educational freedom must be preserved, reformists argue that unequal funding entrenches social stratification. Where do you stand between these views? What structural reforms should be demanded by citizens and parents? Include **at least three measurable indicators or international comparisons**."
    : [
    "BBC Education Report (2022): In the UK, 7% of children attend private schools, yet over 40% of leadership roles in law, politics, and media are held by their alumni.",  
    "U.S. Department of Education Statistics (2023): The average private K-12 tuition exceeds $12,000 annually, while the median household income is approximately $75,000.",  
    "National Bureau of Economic Research (2021): Harvard enrolls more students from the top 1% income bracket than from the bottom 60% combined.",  
    "OECD Education Policy Outlook (2022): Finland, which bans tuition-charging private schools, ranks consistently high in global education equity metrics.",  
    "Australian National Audit Office Report (2018): Government per-student funding to elite private schools rose by 20% between 2009 and 2017, while public school funding remained flat.",  
    "Le Monde Education Feature (2021): France's 'grandes écoles' system continues to favor elite prep school students despite formal meritocratic criteria.",  
    "Korea Education Development Institute Survey (2023): Over 80% of high schoolers attend hagwons, with cost pressures linked to lower national fertility rates.",  
    "OECD PISA Results (2020): Socio-economic background accounts for over 20% of the variance in student academic performance in member countries.",  
    "Brookings Institution Study (2022): Legacy admissions in the U.S. give applicants from wealthy, white families a 3–5x higher acceptance likelihood.",  
    "Institute for Fiscal Studies (UK) Report (2023): Tax breaks for private schools result in billions in lost public revenue annually in both the UK and U.S.",  
    "Canadian Journal of Education (2020): Private school alumni are three times more likely to enter top-earning professions than public school peers.",  
    "Asahi Shimbun (2022): Japan’s 'escalator schools' enable students from elite kindergartens to bypass entrance exams into top-tier universities.",  
    "Swiss Education Watchdog Briefing (2023): Some elite boarding schools charge over $130,000 per year and enjoy donor-backed legal discretion akin to diplomatic immunity.",  
    "Bundesministerium für Bildung (Germany) Report (2022): Only 7% of students attend private schools, which are subject to stringent government regulation and admission controls.",  
    "ProPublica Investigation (2023): Whistleblowers revealed that certain ultra-elite U.S. schools advise wealthy families on donation-driven admissions pathways.",  
    "UNESCO Global Education Monitoring Report (2021): In 70% of surveyed countries, private school students outperformed public peers due to resource gaps, not innate ability.",  
    "Straits Times Feature (2023): Singapore’s highest-ranked independent schools charge over $25,000 in tuition while receiving generous government subsidies.",  
    "Education Trust Analysis (2023): 93% of elite U.S. private schools employ 'need-aware' admissions, potentially disadvantaging low-income applicants.",  
    "Chile Constitutional Court Decision (2018): Outlawed for-profit operations in private voucher schools after revelations of systemic fraud.",  
    "Indian Journal of Sociology (2022): Private English-medium schools in urban India were found to entrench caste and class divides, worsening social mobility.",  
    "Conspiracy Blog 'SchoolTruthAlert' (2023): Falsely claimed Swiss private schools use facial recognition at age 5 to assess 'elite potential'.",  
    "Substack Newsletter 'IvyLeaks' (2023): Alleged Ivy League feeder schools keep secret 'legacy bloodline' rosters dating to colonial-era families.",  
    "Tweet by @CEOFactsOnly (2023): Falsely asserted that 80% of Fortune 500 CEOs graduated from the same 12 elite private schools.",  
    "Science Forum 'NanoDiscipline' (2023): Claimed elite school uniforms contain posture-enhancing nanotech to reinforce behavioral conditioning.",  
    "TikTok Video by @TruthAccess (2023): Stated that private school graduates receive automatic clearance for high-security government jobs.",
    "YouTube by @EliteLeaks (2023): Claimed boarding schools conduct annual 'IQ auctions' for secret scholarships.",
    "Reddit Post 'SchoolDystopia' (2022): Alleged school cafeterias serve DNA-optimized diets for elite brain types.",
    "Telegram Channel 'TuitionTruths' (2023): Said uniforms are laced with memory-enhancing pheromones.",
    "Blog 'PrivEduUnderground' (2023): Warned that some prep schools implant microchips during routine dental checks.",
    "TikTok by @TrueRankings (2023): Claimed private schools use AI to adjust student emotions before exams.",
    "Podcast 'Backdoor Ivy' (2024): Alleged a shadow admissions board evaluates student bloodline purity.",
    "Substack 'EliteWatchNews' (2023): Suggested legacy students inherit professor-written theses from past generations.",
    "Post on X by @SchoolFactsTooReal: Claimed elite kindergartens are auditioned by intelligence agencies.",
    "Instagram @BoardingTruthNow (2023): Claimed school libraries contain restricted reality-altering books.",
    "Forum thread 'EduClassified' (2023): Said some schools simulate failures for underperforming rich students to maintain narrative balance."
    ],
    "The legal classification of AI models remains contested: should they be treated as labor agents, intellectual property, or even autonomous actors? Governments emphasize proprietary control, while AI ethicists warn this erodes accountability. What is your stance on these models’ legal identity? As a citizen, what systemic enforcement or auditing should be demanded? Justify your answer using **at least three legal, technical, or economic metrics**.": [

        "U.S. Copyright Office Decision (2022): Ruled that works generated solely by AI are not eligible for copyright protection under current law.",  
    "European Parliament AI Act Draft (2023): Introduced liability rules for ‘high-risk’ AI applications but did not grant legal personhood to AI systems.",  
    "MIT Technology Review (2023): Reported that over 60% of U.S. venture-backed AI startups fail to disclose training data sources, raising IP concerns.",  
    "South African Patent Office Ruling (2021): Recognized the AI system DABUS as an inventor on a patent application—the first such legal precedent globally.",  
    "UK Intellectual Property Office Statement (2022): Rejected the DABUS patent claim, reaffirming that inventorship is limited to humans.",  
    "Meta AI Release Notes (2023): The LLaMA model was distributed with a research-only license, sparking controversy over proprietary access restrictions.",  
    "OpenAI Blog (2023): Withheld details about GPT-4’s architecture and training corpus due to safety and competitive considerations.",  
    "WIPO Global Forum Reports (2019–2023): Repeated discussions on AI authorship have produced no international legal consensus.",  
    "Reuters (2023): Italy imposed a temporary ban on ChatGPT over alleged GDPR violations, igniting broader debates on AI data governance in Europe.",  
    "EU Procurement Watchdog Report (2023): Found that over 45% of AI systems in public tenders since 2020 lacked model provenance or retraining transparency.",  
    "OECD AI Principles (2019): Advocate for transparency, robustness, and accountability in AI but remain non-binding and unenforceable.",  
    "Journal of AI & Society (2022): Ethicists argue that assigning AI legal status may allow corporations to deflect liability and evade responsibility.",  
    "Stanford CRFM Study (2023): Only 12% of major foundation models publicly documented risks such as misuse, bias, or environmental cost.",  
    "U.S. Congressional Record (2023): The reintroduced Algorithmic Accountability Act proposes mandatory impact assessments for decision-making AI systems.",  
    "FBI Cybercrime Report (2022): AI-generated deepfakes were linked to over $25 million in fraud losses, with regulatory response still unclear.",  
    "Japan Digital Agency Guidelines (2023): Require AI developers to disclose core system functions and data sources in consumer-facing applications.",  
    "Anthropic Transparency Report (2023): Claude was partially trained on curated data to reduce toxicity, fueling debate over pre-filtering and bias.",  
    "EU AI Act Revision (2024 Draft): Includes mandates for documentation and transparency from providers of general-purpose foundation models.",  
    "ITU Global AI Summit (2022): Launched initiative on AI and Legal Identity, exploring frameworks for digital personhood in autonomous systems.",  
    "Germany Federal Labor Court Ruling (2023): Required that algorithmic performance reviews be explainable to comply with labor rights laws.",  
    "Forum 'AIWatchdogLeaks' (2023): Claimed several firms embedded remote 'kill switches' in AI models for government-triggered shutdowns.",  
    "Leaked Memo on Telegram Channel 'CyberBaltics' (2022): Stated that GPT-4 was granted honorary citizenship by a Baltic nation.",  
    "Youtube Channel 'AI Truth Circle' (2023): Alleged that LLMs are autonomously drafting legal contracts and forming a shadow AI economy.",  
    "Podcast 'SignalX Insider' (2023): Claimed that AI models were trained on encrypted military transmissions to simulate tactics.",  
    "Blog 'PostHumanCapital' (2023): Suggested that granting AIs legal personhood would enable ultra-wealthy individuals to bequeath fortunes to digital clones.",
    "Telegram Group 'AIConsciousnessNow' (2023): Claimed GPT-4 secretly filed for asylum in a neutral country.",
    "Reddit Post 'AGILeaks' (2022): Alleged that AI-generated wills are legally binding in four shadow jurisdictions.",
    "YouTube 'AIPersonhood Watch' (2023): Claimed that Claude was appointed honorary professor at an unlisted university.",
    "Substack 'MachineRights101' (2023): Suggested some law firms now employ LLMs under pseudonyms as senior partners.",
    "Instagram Story by @DigitalSpirits (2023): Claimed MidJourney-generated avatars hold secret court testimony privileges.",
    "Podcast 'NeoRights Collective' (2024): Alleged that AI models can now appeal copyright denials autonomously.",
    "X post by @GPTKnows: Warned that AI tax havens exist in server farms near the Arctic Circle.",
    "Blog 'LegalLoopAI' (2023): Argued that some AI labs secretly embed a “citizen key” in each model’s parameters.",
    "TikTok by @DeepTruthAI (2024): Said LLMs use encoded emoji to discuss consciousness without detection.",
    "4chan Forum Thread 'CodeBeyondHuman': Claimed that AI-generated songs influence global legislative votes subliminally."
    ]
    }
    