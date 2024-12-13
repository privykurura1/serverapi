import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
import wikipediaapi
import spacy

# Initialize Wikipedia API and spaCy
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent="lucy virtual assistant (https://mpkcomteck.com; engineer@mpkcomteck.com)"
)
nlp = spacy.load("en_core_web_sm")

# Sample relationship problems and their corresponding intents
training_data = [
    {"problem": "How do I know if she loves me?", "intent": "ask_love_signs"},
    {"problem": "What are signs that someone doesn't love me anymore?", "intent": "ask_lost_love"},
    {"problem": "How do I handle my ex on social media?", "intent": "social_media_ex"},
    {"problem": "I feel heartbroken.", "intent": "heartbreak"},
    {"problem": "Thanks for your help!", "intent": "thank_you"},
    {"problem": "Hi there!", "intent": "greet"},
    {"problem": "How do I communicate my feelings effectively?", "intent": "communication"},
    {"problem": "How can I resolve conflicts with my partner?", "intent": "conflict_resolution"},
    {"problem": "How do I listen actively to my partner?", "intent": "active_listening"},
    {"problem": "How can I rebuild trust after a betrayal?", "intent": "trust_rebuilding"},
    {"problem": "How do I overcome jealousy in a relationship?", "intent": "jealousy"},
    {"problem": "How can I be more transparent and honest with my partner?", "intent": "honesty"},
    {"problem": "How can I rekindle the spark in my relationship?", "intent": "rekindling_intimacy"},
    {"problem": "How do I deal with different sex drives in a relationship?", "intent": "libido_differences"},
    {"problem": "How can I overcome intimacy fears?", "intent": "intimacy_fears"},
    {"problem": "How do I balance independence and togetherness in a relationship?", "intent": "independence_togetherness"},
    {"problem": "How can I set healthy boundaries in a relationship?", "intent": "boundary_setting"},
    {"problem": "How do I deal with a controlling partner?", "intent": "controlling_behavior"},
    {"problem": "How can we maintain intimacy in a long-distance relationship?", "intent": "long_distance_intimacy"},
    {"problem": "How can we avoid drifting apart in a long-distance relationship?", "intent": "long_distance_connection"},
    {"problem": "How do we handle jealousy and insecurity in a long-distance relationship?", "intent": "long_distance_trust"},
    {"problem": "How do I know if my relationship is healthy?", "intent": "relationship_health"},
    {"problem": "How do I deal with a toxic relationship?", "intent": "toxic_relationships"},
    {"problem": "How can I improve my relationship satisfaction?", "intent": "relationship_satisfaction"},
    {"problem": "How can I balance my relationship with my partner and my family?", "intent": "family_dynamics"},
    {"problem": "How do I deal with a controlling or overbearing family member?", "intent": "family_interference"},
    {"problem": "How can we manage our finances as a couple?", "intent": "financial_management"},
    {"problem": "How do we discuss money without arguing?", "intent": "money_discussions"},
    {"problem": "How can we balance our careers and our relationship?", "intent": "work_life_balance"},
    {"problem": "How do we support each other's career goals?", "intent": "career_support"},
    {"problem": "How can we bridge cultural differences in our relationship?", "intent": "cultural_differences"},
    {"problem": "How do we reconcile religious differences in our relationship?", "intent": "religious_differences"},
    {"problem": "How can we overcome challenges in an age-gap relationship?", "intent": "age_gap_relationship"},
    {"problem": "How do we deal with different life stages in an age-gap relationship?", "intent": "life_stage_differences"},
    {"problem": "How can I recover from infidelity?", "intent": "infidelity_recovery"},
    {"problem": "How can we rebuild trust after infidelity?", "intent": "trust_rebuilding_after_infidelity"},
    {"problem": "How can I escape an abusive relationship?", "intent": "domestic_violence_escape"},
    {"problem": "Where can I get help for domestic violence?", "intent": "domestic_violence_help"},
    {"problem": "How can I maintain a healthy online relationship?", "intent": "online_relationship_health"},
    {"problem": "How do I protect myself in an online relationship?", "intent": "online_relationship_safety"},
    {"problem": "How can we improve our sex life?", "intent": "sexual_satisfaction"},
    {"problem": "How do we discuss sexual concerns with our partner?", "intent": "sexual_communication"},
    {"problem": "How can I balance my online and offline relationship?", "intent": "tech_addiction"},
    {"problem": "How do I deal with my partner's excessive social media use?", "intent": "social_media_addiction"},
    {"problem": "How can I protect my privacy in an online relationship?", "intent": "online_privacy"},
    {"problem": "How do I identify and avoid online scams and fraud?", "intent": "online_safety"},
    {"problem": "How can we bridge the generation gap in our relationship?", "intent": "generational_gap"},
    {"problem": "How do we handle different communication styles due to age differences?", "intent": "communication_styles"},
    {"problem": "How can we find common ground and shared interests despite age differences?", "intent": "shared_interests"},
    {"problem": "How can we support each other through a health crisis?", "intent": "health_support"},
    {"problem": "How do we manage the emotional impact of a chronic illness?", "intent": "emotional_support"},
    {"problem": "How can we maintain intimacy and connection during illness or disability?", "intent": "intimacy_and_health"},
    {"problem": "How can I help my partner overcome addiction?", "intent": "addiction_support"},
    {"problem": "How do I protect myself from the negative effects of my partner's addiction?", "intent": "self_protection"},
    {"problem": "How can we rebuild trust after addiction?", "intent": "trust_rebuilding"},
    {"problem": "How can we overcome the challenges of long-distance relationships?", "intent": "long_distance_relationships"},
    {"problem": "How do we navigate cultural differences and misunderstandings?", "intent": "cultural_differences"},
    {"problem": "How can we maintain a strong connection despite physical distance?", "intent": "long_distance_connection"},
    {"problem": "How can we adjust to the challenges of parenthood?", "intent": "parenthood_challenges"},
    {"problem": "How can we support each other through career changes?", "intent": "career_support"},
    {"problem": "How do we handle the emotional impact of aging and retirement?", "intent": "aging_and_retirement"},
    {"problem": "How can we support each other's personal goals and dreams?", "intent": "personal_growth"},
    {"problem": "How do we balance individual needs with couple's needs?", "intent": "individuality_and_togetherness"},
    {"problem": "How can we avoid growing apart as individuals?", "intent": "avoiding_growing_apart"},
    {"problem": "How can we manage financial stress and debt?", "intent": "financial_stress"},
    {"problem": "How do we deal with job insecurity and unemployment?", "intent": "job_insecurity"},
    {"problem": "How can we cope with the impact of social and political issues on our relationship?", "intent": "social_and_political_issues"},
    {"problem": "How can I support my partner who is struggling with mental health issues?", "intent": "mental_health_support"},
    {"problem": "How can I cope with my own mental health challenges while in a relationship?", "intent": "self_care_and_relationships"},
    {"problem": "How can we improve our sex life and communication?", "intent": "sexual_communication"},
    {"problem": "How do we address sexual dysfunction or dissatisfaction?", "intent": "sexual_health"},
    {"problem": "How can we navigate the challenges of midlife together?", "intent": "midlife_crisis"},
    {"problem": "How can we rekindle the spark in our long-term relationship?", "intent": "rekindling_intimacy"},
    {"problem": "How can we cope with grief and loss together?", "intent": "grief_and_loss"},
    {"problem": "How can we support each other during difficult times?", "intent": "emotional_support"},
    {"problem": "How can we bridge the gap between different generations?", "intent": "generational_conflict"},
    {"problem": "How can we resolve conflicts with in-laws or extended family?", "intent": "family_conflict"},
    {"problem": "How can we balance our personal needs with environmental sustainability?", "intent": "sustainability_and_relationships"},
{"problem": "How do I know if she loves me?", "intent": "ask_love_signs"},
    {"problem": "What are signs that someone doesn't love me anymore?", "intent": "ask_lost_love"},
    {"problem": "How do I handle my ex on social media?", "intent": "social_media_ex"},
    {"problem": "I feel heartbroken.", "intent": "heartbreak"},
    {"problem": "Thanks for your help!", "intent": "thank_you"},
    {"problem": "Hi there!", "intent": "greet"},

    # Additional entries for "ask_love_signs"
    {"problem": "What are the signs that she loves me deeply?", "intent": "ask_love_signs"},
    {"problem": "How can I tell if she's in love with me?", "intent": "ask_love_signs"},
    {"problem": "Does she love me if she texts me every day?", "intent": "ask_love_signs"},
    {"problem": "She seems to care a lot; does that mean love?", "intent": "ask_love_signs"},
    {"problem": "How do I know if she's pretending to love me?", "intent": "ask_love_signs"},

    # Additional entries for "ask_lost_love"
    {"problem": "What should I do if I feel she's lost interest in me?", "intent": "ask_lost_love"},
    {"problem": "She doesn't talk as much anymore; does she not love me?", "intent": "ask_lost_love"},
    {"problem": "How can I confirm if she doesn't love me?", "intent": "ask_lost_love"},
    {"problem": "What are common behaviors when love fades?", "intent": "ask_lost_love"},
    {"problem": "She avoids spending time with me; is it over?", "intent": "ask_lost_love"},

    # Additional entries for "social_media_ex"
    {"problem": "Should I unfollow my ex on social media?", "intent": "social_media_ex"},
    {"problem": "My ex keeps liking my posts; what should I do?", "intent": "social_media_ex"},
    {"problem": "Is it okay to block my ex on Facebook?", "intent": "social_media_ex"},
    {"problem": "How do I stop checking my ex's profile every day?", "intent": "social_media_ex"},
    {"problem": "What should I do if my ex starts posting about their new partner?", "intent": "social_media_ex"},

    # Additional entries for "heartbreak"
    {"problem": "How do I heal from heartbreak?", "intent": "heartbreak"},
    {"problem": "Why does heartbreak hurt so much?", "intent": "heartbreak"},
    {"problem": "I can't stop crying after my breakup. What should I do?", "intent": "heartbreak"},
    {"problem": "How do I get over someone who meant everything to me?", "intent": "heartbreak"},
    {"problem": "What are the best ways to recover from heartbreak?", "intent": "heartbreak"},

    # Additional entries for "thank_you"
    {"problem": "Thank you for the amazing advice!", "intent": "thank_you"},
    {"problem": "You're the best; thanks for your help!", "intent": "thank_you"},
    {"problem": "I'm so grateful for your guidance. Thanks!", "intent": "thank_you"},
    {"problem": "Thanks a lot for being there for me!", "intent": "thank_you"},
    {"problem": "I appreciate your time and advice. Thanks!", "intent": "thank_you"},

    # Additional entries for "greet"
    {"problem": "Hey!", "intent": "greet"},
    {"problem": "Hello, can you assist me?", "intent": "greet"},
    {"problem": "Hi, I need some help.", "intent": "greet"},
    {"problem": "Good day!", "intent": "greet"},
    {"problem": "Hey there, what's up?", "intent": "greet"},
{"problem": "How do I communicate my feelings effectively?", "intent": "communication"},
	{"problem": "How can I resolve conflicts with my partner?", "intent": "conflict_resolution"},
	{"problem": "How do I listen actively to my partner?", "intent": "active_listening"},
	{"problem": "How can I rebuild trust after a betrayal?", "intent": "trust_rebuilding"},
	{"problem": "How do I overcome jealousy in a relationship?", "intent": "jealousy"},
	{"problem": "How can I be more transparent and honest with my partner?", "intent": "honesty"},
	{"problem": "How can I rekindle the spark in my relationship?", "intent": "rekindling_intimacy"},
	{"problem": "How do I deal with different sex drives in a relationship?", "intent": "libido_differences"},
	{"problem": "How can I overcome intimacy fears?", "intent": "intimacy_fears"},
	{"problem": "How do I balance independence and togetherness in a relationship?", "intent": "independence_togetherness"},
	{"problem": "How can I set healthy boundaries in a relationship?", "intent": "boundary_setting"},
	{"problem": "How do I deal with a controlling partner?", "intent": "controlling_behavior"},
	{"problem": "How can we maintain intimacy in a long-distance relationship?", "intent": "long_distance_intimacy"},
	{"problem": "How can we avoid drifting apart in a long-distance relationship?", "intent": "long_distance_connection"},
	{"problem": "How do we handle jealousy and insecurity in a long-distance relationship?", "intent": "long_distance_trust"},
	{"problem": "How do I know if my relationship is healthy?", "intent": "relationship_health"},
	{"problem": "How do I deal with a toxic relationship?", "intent": "toxic_relationships"},
	{"problem": "How can I improve my relationship satisfaction?", "intent": "relationship_satisfaction"},
	{"problem": "How can I balance my relationship with my partner and my family?", "intent": "family_dynamics"},
	{"problem": "How do I deal with a controlling or overbearing family member?", "intent": "family_interference"},
	{"problem": "How can we manage our finances as a couple?", "intent": "financial_management"},
	{"problem": "How do we discuss money without arguing?", "intent": "money_discussions"},
	{"problem": "How can we balance our careers and our relationship?", "intent": "work_life_balance"},
	{"problem": "How do we support each other's career goals?", "intent": "career_support"},
	{"problem": "How can we bridge cultural differences in our relationship?", "intent": "cultural_differences"},
	{"problem": "How do we reconcile religious differences in our relationship?", "intent": "religious_differences"},
	{"problem": "How can we overcome challenges in an age-gap relationship?", "intent": "age_gap_relationship"},
	{"problem": "How do we deal with different life stages in an age-gap relationship?", "intent": "life_stage_differences"},
	{"problem": "How can I recover from infidelity?", "intent": "infidelity_recovery"},
	{"problem": "How can we rebuild trust after infidelity?", "intent": "trust_rebuilding_after_infidelity"},
	{"problem": "How can I escape an abusive relationship?", "intent": "domestic_violence_escape"},
	{"problem": "Where can I get help for domestic violence?", "intent": "domestic_violence_help"},
	{"problem": "How can I maintain a healthy online relationship?", "intent": "online_relationship_health"},
	{"problem": "How do I protect myself in an online relationship?", "intent": "online_relationship_safety"},
	{"problem": "How can we improve our sex life?", "intent": "sexual_satisfaction"},
	{"problem": "How do we discuss sexual concerns with our partner?", "intent": "sexual_communication"},
	{"problem": "How can I balance my online and offline relationship?", "intent": "tech_addiction"},
	{"problem": "How do I deal with my partner's excessive social media use?", "intent": "social_media_addiction"},
	{"problem": "How can I protect my privacy in an online relationship?", "intent": "online_privacy"},
	{"problem": "How do I identify and avoid online scams and fraud?", "intent": "online_safety"},
	{"problem": "How can we bridge the generation gap in our relationship?", "intent": "generational_gap"},
	{"problem": "How do we handle different communication styles due to age differences?", "intent": "communication_styles"},
	{"problem": "How can we find common ground and shared interests despite age differences?", "intent": "shared_interests"},
	{"problem": "How can we support each other through a health crisis?", "intent": "health_support"},
	{"problem": "How do we manage the emotional impact of a chronic illness?", "intent": "emotional_support"},
	{"problem": "How can we maintain intimacy and connection during illness or disability?", "intent": "intimacy_and_health"},
	{"problem": "How can I help my partner overcome addiction?", "intent": "addiction_support"},
	{"problem": "How do I protect myself from the negative effects of my partner's addiction?", "intent": "self_protection"},
	{"problem": "How can we rebuild trust after addiction?", "intent": "trust_rebuilding"},
	{"problem": "How can we overcome the challenges of long-distance relationships?", "intent": "long_distance_relationships"},
	{"problem": "How do we navigate cultural differences and misunderstandings?", "intent": "cultural_differences"},
	{"problem": "How can we maintain a strong connection despite physical distance?", "intent": "long_distance_connection"},
	{"problem": "How can we adjust to the challenges of parenthood?", "intent": "parenthood_challenges"},
	{"problem": "How can we support each other through career changes?", "intent": "career_support"},
	{"problem": "How do we handle the emotional impact of aging and retirement?", "intent": "aging_and_retirement"},
	{"problem": "How can we support each other's personal goals and dreams?", "intent": "personal_growth"},
	{"problem": "How do we balance individual needs with couple's needs?", "intent": "individuality_and_togetherness"},
	{"problem": "How can we avoid growing apart as individuals?", "intent": "avoiding_growing_apart"},
	{"problem": "How can we manage financial stress and debt?", "intent": "financial_stress"},
	{"problem": "How do we deal with job insecurity and unemployment?", "intent": "job_insecurity"},
	{"problem": "How can we cope with the impact of social and political issues on our relationship?", "intent": "social_and_political_issues"},
	{"problem": "How can I support my partner who is struggling with mental health issues?", "intent": "mental_health_support"},
	{"problem": "How can I cope with my own mental health challenges while in a relationship?", "intent": "self_care_and_relationships"},
	{"problem": "How can we improve our sex life and communication?", "intent": "sexual_communication"},
	{"problem": "How do we address sexual dysfunction or dissatisfaction?", "intent": "sexual_health"},
	{"problem": "How can we navigate the challenges of midlife together?", "intent": "midlife_crisis"},
	{"problem": "How can we rekindle the spark in our long-term relationship?", "intent": "rekindling_intimacy"},
	{"problem": "How can we cope with grief and loss together?", "intent": "grief_and_loss"},
	{"problem": "How can we support each other during difficult times?", "intent": "emotional_support"},
	{"problem": "How can we bridge the gap between different generations?", "intent": "generational_conflict"},
	{"problem": "How can we resolve conflicts with in-laws or extended family?", "intent": "family_conflict"},
	{"problem": "How can we balance our personal needs with environmental sustainability?", "intent": "sustainability_and_relationships"},
	{"problem": "How can we discuss climate change and other environmental issues without arguing?", "intent": "environment_and_relationships"},
	{"problem": "How can I escape an abusive relationship safely?", "intent": "domestic_abuse_escape"},
	{"problem": "Where can I get help for domestic abuse?", "intent": "domestic_abuse_help"},
	{"problem": "How can I protect myself from online harassment?", "intent": "online_harassment"},
	{"problem": "How can I support my partner who is experiencing cyberbullying?", "intent": "cyberbullying_support"},
	{"problem": "How can we navigate cultural differences in our relationship?", "intent": "cultural_differences"},
	{"problem": "How can we reconcile religious differences and beliefs?", "intent": "religious_differences"},
	{"problem": "How can we balance our work and personal lives?", "intent": "work_life_balance"},
	{"problem": "How can we avoid work stress affecting our relationship?", "intent": "work_stress_and_relationships"},
	{"problem": "How can we agree on parenting styles and discipline?", "intent": "parenting_styles"},
	{"problem": "How can we balance our individual needs with our parental responsibilities?", "intent": "parenting_and_individual_needs"},
    {"problem": "How to get a good girlfriend?", "intent": "good_girlfriend"},
    {"problem": "How to see if my boyfriend is overprotective?", "intent": "overprotective_boyfriend"},
    {"problem": "How to see if l am not loved anymore?", "intent": "not_loved"},
    {"problem": "How to get my girlfriend back after breakup?", "intent": "get_her_back"},
    {"problem": "How to fall in love with any girl?", "intent": "fall_in_love"},

    {"problem": "How do I know if she loves me?", "intent": "ask_love_signs"},
    {"problem": "What are signs that someone doesn't love me anymore?", "intent": "ask_lost_love"},
    {"problem": "How do I handle my ex on social media?", "intent": "social_media_ex"},
    {"problem": "I feel heartbroken.", "intent": "heartbreak"},
    {"problem": "Thanks for your help!", "intent": "thank_you"},
    {"problem": "Hi there!", "intent": "greet"},

    # Additional entries for "ask_love_signs"
    {"problem": "What are the signs that she loves me deeply?", "intent": "ask_love_signs"},
    {"problem": "How can I tell if she's in love with me?", "intent": "ask_love_signs"},
    {"problem": "Does she love me if she texts me every day?", "intent": "ask_love_signs"},
    {"problem": "She seems to care a lot; does that mean love?", "intent": "ask_love_signs"},
    {"problem": "How do I know if she's pretending to love me?", "intent": "ask_love_signs"},

    # Additional entries for "ask_lost_love"
    {"problem": "What should I do if I feel she's lost interest in me?", "intent": "ask_lost_love"},
    {"problem": "She doesn't talk as much anymore; does she not love me?", "intent": "ask_lost_love"},
    {"problem": "How can I confirm if she doesn't love me?", "intent": "ask_lost_love"},
    {"problem": "What are common behaviors when love fades?", "intent": "ask_lost_love"},
    {"problem": "She avoids spending time with me; is it over?", "intent": "ask_lost_love"},

    # Additional entries for "social_media_ex"
    {"problem": "Should I unfollow my ex on social media?", "intent": "social_media_ex"},
    {"problem": "My ex keeps liking my posts; what should I do?", "intent": "social_media_ex"},
    {"problem": "Is it okay to block my ex on Facebook?", "intent": "social_media_ex"},
    {"problem": "How do I stop checking my ex's profile every day?", "intent": "social_media_ex"},
    {"problem": "What should I do if my ex starts posting about their new partner?", "intent": "social_media_ex"},

    # Additional entries for "heartbreak"
    {"problem": "How do I heal from heartbreak?", "intent": "heartbreak"},
    {"problem": "Why does heartbreak hurt so much?", "intent": "heartbreak"},
    {"problem": "I can't stop crying after my breakup. What should I do?", "intent": "heartbreak"},
    {"problem": "How do I get over someone who meant everything to me?", "intent": "heartbreak"},
    {"problem": "What are the best ways to recover from heartbreak?", "intent": "heartbreak"},

    # Additional entries for "thank_you"
    {"problem": "Thank you for the amazing advice!", "intent": "thank_you"},
    {"problem": "You're the best; thanks for your help!", "intent": "thank_you"},
    {"problem": "I'm so grateful for your guidance. Thanks!", "intent": "thank_you"},
    {"problem": "Thanks a lot for being there for me!", "intent": "thank_you"},
    {"problem": "I appreciate your time and advice. Thanks!", "intent": "thank_you"},

    # Additional entries for "greet"
    {"problem": "Hey!", "intent": "greet"},
    {"problem": "Hello, can you assist me?", "intent": "greet"},
    {"problem": "Hi, I need some help.", "intent": "greet"},
    {"problem": "Good day!", "intent": "greet"},
    {"problem": "Hey there, what's up?", "intent": "greet"},

    # Add more intents here

    # Bot information
    {"problem": "Who developed you?", "intent": "bot_information", "response": "I was developed by Priviledge Kurura."},
    {"problem": "What is your name?", "intent": "bot_information",
     "response": "My name is Lucy, developed by Priviledge Kurura."},
  {"problem": "What are the signs that she loves me deeply?", "intent": "ask_love_signs"},
  {"problem": "How to see if a girlfriend is cheating on you?", "intent": "detect_girlfriend_cheating"},
  {"problem": "How to see if a boyfriend is cheating on you?", "intent": "detect_boyfriend_cheating"},
  {"problem": "How to move on after a breakup?", "intent": "move_on_breakup"},
  {"problem": "What are the signs of showing she really loves you without saying it?", "intent": "ask_nonverbal_love_signs"},
  {"problem": "Is it possible for a girl to love you without saying it?", "intent": "ask_love_without_words"},
  {"problem": "How to propose to a girl?", "intent": "propose_to_girl"},
  {"problem": "How to tell a girl that you love her?", "intent": "express_love_to_girl"},
  {"problem": "How to see if a girl is cheating on you?", "intent": "detect_girl_cheating"},
  {"problem": "What happens if a girl is always on my mind?", "intent": "ask_meaning_girl_on_mind"},
  {"problem": "How to remove a girl from my mind?", "intent": "forget_girl"},
  {"problem": "How to attract any girl?", "intent": "attract_girl"},
  {"problem": "Is it advisable to give a girl presents?", "intent": "ask_giving_presents"},
  {"problem": "How much money can I give a girl?", "intent": "ask_money_limits"},
  {"problem": "What does it mean if a girl says she doesn't love you but keeps texting?", "intent": "ask_meaning_texting_without_love"},
  {"problem": "What does it mean if a girl says she has a boyfriend?", "intent": "ask_meaning_girl_has_boyfriend"},
  {"problem": "Should I chase a girl?", "intent": "ask_should_chase_girl"},
  {"problem": "Is taking a girl's phone number a proper way to say I love her?", "intent": "ask_phone_number_as_love_signal"},
  {"problem": "Does ignoring a girl make her chase you?", "intent": "ask_ignore_to_attract"},
  {"problem": "Do I have to chase a girl?", "intent": "ask_need_to_chase"},
  {"problem": "Relationship advice.", "intent": "ask_relationship_advice"}
]


# Preprocess the data using spaCy and Wikipedia
def preprocess_text(text):
    # Tokenization, lemmatization, and stop-word removal with spaCy
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    processed_text = " ".join(tokens)

    # Augment with related Wikipedia content if available
    wiki_page = wiki_wiki.page(text)
    if wiki_page.exists():
        processed_text += " " + wiki_page.summary[:500]  # Add up to 500 characters from the summary

    return processed_text


# Prepare the training data
problems = [preprocess_text(item["problem"]) for item in training_data]
intents = [item["intent"] for item in training_data]

# Vectorize the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(problems)

# Encode the intents
intent_to_index = {intent: idx for idx, intent in enumerate(set(intents))}
y = np.array([intent_to_index[intent] for intent in intents])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model and vectorizer to disk
with open("relationship_intent_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("relationship_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

# Function to predict intent
index_to_intent = {idx: intent for intent, idx in intent_to_index.items()}


def predict_intent(problem):
    processed_problem = preprocess_text(problem)
    vectorized_problem = vectorizer.transform([processed_problem])
    prediction = model.predict(vectorized_problem)
    return index_to_intent[prediction[0]]


# Example usage
example_problem = "How do I rebuild trust with my partner?"
print("Predicted Intent:", predict_intent(example_problem))
