import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
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
    {"problem": "How do I handle relationship issues?", "intent": "ask_relationship_advice"},
    {"problem": "How do I know if she loves me?", "intent": "ask_love_signs"},
    {"problem": "What are signs that someone doesn't love me anymore?", "intent": "ask_lost_love"},
    {"problem": "How do I handle my ex on social media?", "intent": "social_media_ex"},
    {"problem": "I feel heartbroken.", "intent": "heartbreak"},
    {"problem": "Thanks for your help!", "intent": "thank_you"},
    {"problem": "Hi there!", "intent": "greet"},
    {"problem": "How can I communicate better?", "intent": "communication"},
    {"problem": "How do I resolve conflicts in my relationship?", "intent": "conflict_resolution"},
    {"problem": "How can I listen better to my partner?", "intent": "active_listening"},
    {"problem": "How can I rebuild trust?", "intent": "trust_rebuilding"},
    {"problem": "How do I deal with jealousy?", "intent": "jealousy"},
    {"problem": "How important is honesty in a relationship?", "intent": "honesty"},
    {"problem": "How do I rekindle intimacy?", "intent": "rekindling_intimacy"},
    {"problem": "What if we have different sex drives?", "intent": "libido_differences"},
    {"problem": "I'm afraid of intimacy. What should I do?", "intent": "intimacy_fears"},
    {"problem": "How do we balance independence and togetherness?", "intent": "independence_togetherness"},
    {"problem": "How can we set healthy boundaries?", "intent": "boundary_setting"},
    {"problem": "My partner is controlling. How do I deal with this?", "intent": "controlling_behavior"},
    {"problem": "How do I maintain intimacy in a long-distance relationship?", "intent": "long_distance_intimacy"},
    {"problem": "How can we stay connected in a long-distance relationship?", "intent": "long_distance_connection"},
    {"problem": "How do I rebuild trust after a long-distance breakup?", "intent": "long_distance_trust"},
    {"problem": "What makes a relationship healthy?", "intent": "relationship_health"},
    {"problem": "How do I know if my relationship is toxic?", "intent": "toxic_relationships"},
    {"problem": "How can I improve my relationship satisfaction?", "intent": "relationship_satisfaction"},
    {"problem": "How do I balance my relationship and family?", "intent": "family_dynamics"},
    {"problem": "How do I handle family interference in my relationship?", "intent": "family_interference"},
    {"problem": "How should we manage finances together?", "intent": "financial_management"},
    {"problem": "How do we talk about money without arguing?", "intent": "money_discussions"},
    {"problem": "How do I balance work and my relationship?", "intent": "work_life_balance"},
    {"problem": "How do I support my partner's career?", "intent": "career_support"},
    {"problem": "How do I deal with cultural differences?", "intent": "cultural_differences"},
    {"problem": "What should we do about religious differences?", "intent": "religious_differences"},
    {"problem": "How do we handle an age-gap relationship?", "intent": "age_gap_relationship"},
    {"problem": "How do we handle different life stages?", "intent": "life_stage_differences"},
    {"problem": "How do I recover from infidelity?", "intent": "infidelity_recovery"},
    {"problem": "How can trust be rebuilt after infidelity?", "intent": "trust_rebuilding_after_infidelity"},
    {"problem": "What should I do if I'm in an abusive relationship?", "intent": "domestic_violence_escape"},
    {"problem": "How can I get help in an abusive relationship?", "intent": "domestic_violence_help"},
    {"problem": "How do I maintain an online relationship?", "intent": "online_relationship_health"},
    {"problem": "How do I protect myself in an online relationship?", "intent": "online_relationship_safety"},
    {"problem": "How do I improve sexual satisfaction?", "intent": "sexual_satisfaction"},
    {"problem": "How do I communicate better sexually?", "intent": "sexual_communication"},
    {"problem": "How do I deal with tech addiction?", "intent": "tech_addiction"},
    {"problem": "How do I handle social media addiction in my relationship?", "intent": "social_media_addiction"},
    {"problem": "How do I protect my online privacy?", "intent": "online_privacy"},
    {"problem": "How do I stay safe online?", "intent": "online_safety"},
    {"problem": "How do I handle a generational gap?", "intent": "generational_gap"},
    {"problem": "How do we manage different communication styles?", "intent": "communication_styles"},
    {"problem": "How do we find common interests?", "intent": "shared_interests"},
    {"problem": "How can I support my partner during a health crisis?", "intent": "health_support"},
    {"problem": "How do I provide emotional support?", "intent": "emotional_support"},
    {"problem": "How do I maintain intimacy during illness?", "intent": "intimacy_and_health"},
    {"problem": "How do I support my partner through addiction?", "intent": "addiction_support"},
    {"problem": "How do I protect myself from a partner's addiction?", "intent": "self_protection"},
    {"problem": "How do I handle a long-distance relationship?", "intent": "long_distance_relationships"},
    {"problem": "How do I adjust to parenthood?", "intent": "parenthood_challenges"},
    {"problem": "How do I manage aging and retirement together?", "intent": "aging_and_retirement"},
    {"problem": "How do I support my partner's personal growth?", "intent": "personal_growth"},
    {"problem": "How do I balance individuality and togetherness?", "intent": "individuality_and_togetherness"},
    {"problem": "How do I avoid growing apart?", "intent": "avoiding_growing_apart"},
    {"problem": "How do we manage financial stress?", "intent": "financial_stress"},
    {"problem": "How do I handle job insecurity in my relationship?", "intent": "job_insecurity"},
    {"problem": "How do I discuss social and political issues in my relationship?", "intent": "social_and_political_issues"},
    {"problem": "How do I support my partner's mental health?", "intent": "mental_health_support"},
    {"problem": "How do I take care of myself in a relationship?", "intent": "self_care_and_relationships"},
    {"problem": "How do I handle sexual dysfunction?", "intent": "sexual_health"},
    {"problem": "How do we cope with a midlife crisis?", "intent": "midlife_crisis"},
    {"problem": "How do we cope with grief?", "intent": "grief_and_loss"},
    {"problem": "How do I resolve generational conflict?", "intent": "generational_conflict"},
    {"problem": "How do I handle family conflict?", "intent": "family_conflict"},
    {"problem": "How do we incorporate sustainability into our relationship?", "intent": "sustainability_and_relationships"},
    {"problem": "How do we discuss environmental issues in a relationship?", "intent": "environment_and_relationships"},
    {"problem": "How do I escape domestic abuse?", "intent": "domestic_abuse_escape"},
    {"problem": "How can I get help in an abusive situation?", "intent": "domestic_abuse_help"},
    {"problem": "How do I deal with online harassment?", "intent": "online_harassment"},
    {"problem": "How do I support my partner with cyberbullying?", "intent": "cyberbullying_support"},
    {"problem": "How do I attract a good girlfriend?", "intent": "good_girlfriend"},
    {"problem": "How do I handle an overprotective boyfriend?", "intent": "overprotective_boyfriend"},
    {"problem": "What if I feel unloved?", "intent": "not_loved"},
    {"problem": "How do I win her back?", "intent": "get_her_back"},
    {"problem": "How do I fall in love?", "intent": "fall_in_love"},
    # Relationship advice
    {"problem": "Can you give me some relationship advice?", "intent": "ask_relationship_advice"},
    {"problem": "What makes a healthy relationship?", "intent": "ask_relationship_advice"},
    {"problem": "How can I improve my relationship?", "intent": "ask_relationship_advice"},
    {"problem": "What are the key elements of a good relationship?", "intent": "ask_relationship_advice"},

    # Love signs
    {"problem": "How do I know if she loves me?", "intent": "ask_love_signs"},
    {"problem": "What are the signs that a girl likes you?", "intent": "ask_love_signs"},
    {"problem": "How do I tell if she’s interested in me?", "intent": "ask_love_signs"},
    {"problem": "What are the signs of true love?", "intent": "ask_love_signs"},

    # Lost love
    {"problem": "How do I get over someone who doesn’t care anymore?", "intent": "ask_lost_love"},
    {"problem": "What should I do if my partner no longer loves me?", "intent": "ask_lost_love"},
    {"problem": "What if someone stops showing affection?", "intent": "ask_lost_love"},
    {"problem": "How do I cope with unrequited love?", "intent": "ask_lost_love"},

    # Social media and exes
    {"problem": "Should I follow my ex on social media?", "intent": "social_media_ex"},
    {"problem": "Is it okay to stay friends with my ex online?", "intent": "social_media_ex"},
    {"problem": "How do I handle my ex on social media?", "intent": "social_media_ex"},
    {"problem": "How do I manage social media after a breakup?", "intent": "social_media_ex"},

    # Heartbreak
    {"problem": "How do I deal with a broken heart?", "intent": "heartbreak"},
    {"problem": "I’m feeling heartbroken, what should I do?", "intent": "heartbreak"},
    {"problem": "How can I get over heartbreak?", "intent": "heartbreak"},
    {"problem": "How long does it take to heal from heartbreak?", "intent": "heartbreak"},

    # Thank you
    {"problem": "Thanks for the help!", "intent": "thank_you"},
    {"problem": "I appreciate it!", "intent": "thank_you"},
    {"problem": "Thank you so much!", "intent": "thank_you"},
    {"problem": "Thanks for the advice!", "intent": "thank_you"},

    # Greetings
    {"problem": "Hi!", "intent": "greet"},
    {"problem": "Hello, how are you?", "intent": "greet"},
    {"problem": "Good morning!", "intent": "greet"},
    {"problem": "Hey there!", "intent": "greet"},

    # Communication in relationships
    {"problem": "How can I communicate better with my partner?", "intent": "communication"},
    {"problem": "What’s the best way to talk to my partner?", "intent": "communication"},
    {"problem": "How do I have an honest conversation?", "intent": "communication"},
    {"problem": "What’s the key to effective communication?", "intent": "communication"},

    # Conflict resolution
    {"problem": "How do we resolve arguments in a relationship?", "intent": "conflict_resolution"},
    {"problem": "What should we do when we disagree?", "intent": "conflict_resolution"},
    {"problem": "How can we compromise during fights?", "intent": "conflict_resolution"},
    {"problem": "How do we handle disagreements peacefully?", "intent": "conflict_resolution"},

    # Active listening
    {"problem": "How do I listen better in a relationship?", "intent": "active_listening"},
    {"problem": "What is active listening in communication?", "intent": "active_listening"},
    {"problem": "How can I be a better listener?", "intent": "active_listening"},
    {"problem": "Why is active listening important?", "intent": "active_listening"},

    # Trust rebuilding
    {"problem": "How can I rebuild trust in my relationship?", "intent": "trust_rebuilding"},
    {"problem": "What should I do to regain trust after a fight?", "intent": "trust_rebuilding"},
    {"problem": "How can trust be restored after betrayal?", "intent": "trust_rebuilding"},
    {"problem": "How do I regain trust after dishonesty?", "intent": "trust_rebuilding"},

    # Jealousy
    {"problem": "How do I deal with jealousy in my relationship?", "intent": "jealousy"},
    {"problem": "What should I do if I feel jealous?", "intent": "jealousy"},
    {"problem": "How do I overcome jealousy?", "intent": "jealousy"},
    {"problem": "How can I trust my partner more and reduce jealousy?", "intent": "jealousy"},

    # Honesty in relationships
    {"problem": "How do I stay honest in a relationship?", "intent": "honesty"},
    {"problem": "Is honesty really the best policy in a relationship?", "intent": "honesty"},
    {"problem": "Why is honesty important in love?", "intent": "honesty"},
    {"problem": "What happens if honesty is lost in a relationship?", "intent": "honesty"},

    # Rekindling intimacy
    {"problem": "How can I bring intimacy back into my relationship?", "intent": "rekindling_intimacy"},
    {"problem": "What do I do if we’ve lost intimacy?", "intent": "rekindling_intimacy"},
    {"problem": "How do I make our relationship more intimate again?", "intent": "rekindling_intimacy"},
    {"problem": "What can I do to rekindle the romance?", "intent": "rekindling_intimacy"},

    # Libido differences
    {"problem": "What should I do if my partner’s sex drive is different than mine?", "intent": "libido_differences"},
    {"problem": "How do we deal with differences in libido?", "intent": "libido_differences"},
    {"problem": "How can I handle sexual differences in a relationship?", "intent": "libido_differences"},
    {"problem": "How do I talk about sexual differences with my partner?", "intent": "libido_differences"},

    # Intimacy fears
    {"problem": "How do I overcome my fear of intimacy?", "intent": "intimacy_fears"},
    {"problem": "What do I do if I’m afraid of getting close to my partner?", "intent": "intimacy_fears"},
    {"problem": "How can we rebuild intimacy after fear?", "intent": "intimacy_fears"},
    {"problem": "How do I face my fear of emotional closeness?", "intent": "intimacy_fears"},

    # Independence vs. togetherness
    {"problem": "How can we balance independence and time together?", "intent": "independence_togetherness"},
    {"problem": "Is it okay to spend time apart in a relationship?", "intent": "independence_togetherness"},
    {"problem": "How do I maintain my own identity in a relationship?", "intent": "independence_togetherness"},
    {"problem": "How do I create a healthy balance between personal space and time together?",
     "intent": "independence_togetherness"},

    # Boundary setting
    {"problem": "What are healthy relationship boundaries?", "intent": "boundary_setting"},
    {"problem": "How do I set boundaries with my partner?", "intent": "boundary_setting"},
    {"problem": "Why are boundaries important in relationships?", "intent": "boundary_setting"},
    {"problem": "How do I establish clear boundaries?", "intent": "boundary_setting"},

    # Controlling behavior
    {"problem": "How do I handle a controlling partner?", "intent": "controlling_behavior"},
    {"problem": "What do I do if my partner is too possessive?", "intent": "controlling_behavior"},
    {"problem": "How do I communicate with someone who’s controlling?", "intent": "controlling_behavior"},
    {"problem": "How can I set limits with a controlling partner?", "intent": "controlling_behavior"},

    # Long-distance intimacy
    {"problem": "How do we keep intimacy alive in a long-distance relationship?", "intent": "long_distance_intimacy"},
    {"problem": "How can I stay emotionally connected when we’re far apart?", "intent": "long_distance_intimacy"},
    {"problem": "What can we do to maintain intimacy while apart?", "intent": "long_distance_intimacy"},
    {"problem": "How do we maintain intimacy in a long-distance relationship?", "intent": "long_distance_intimacy"},

    # Long-distance connection
    {"problem": "How do we prevent growing apart in a long-distance relationship?",
     "intent": "long_distance_connection"},
    {"problem": "What can we do to stay connected in a long-distance relationship?",
     "intent": "long_distance_connection"},
    {"problem": "How do we keep a relationship strong when apart?", "intent": "long_distance_connection"},
    {"problem": "How do we stay close emotionally despite the distance?", "intent": "long_distance_connection"},

    # Long-distance trust
    {"problem": "How do we build trust in a long-distance relationship?", "intent": "long_distance_trust"},
    {"problem": "What should we do to maintain trust when far apart?", "intent": "long_distance_trust"},
    {"problem": "How do I trust my partner in a long-distance relationship?", "intent": "long_distance_trust"},
    {"problem": "How do I overcome trust issues in a long-distance relationship?", "intent": "long_distance_trust"},

    # Healthy relationship
    {"problem": "What makes a relationship healthy?", "intent": "relationship_health"},
    {"problem": "How do I keep my relationship strong?", "intent": "relationship_health"},
    {"problem": "How do I know if my relationship is healthy?", "intent": "relationship_health"},
    {"problem": "What are the signs of a healthy relationship?", "intent": "relationship_health"},

    # Toxic relationships
    {"problem": "How do I know if I’m in a toxic relationship?", "intent": "toxic_relationships"},
    {"problem": "What should I do if I’m in a toxic relationship?", "intent": "toxic_relationships"},
    {"problem": "How do I escape a toxic relationship?", "intent": "toxic_relationships"},
    {"problem": "What are the signs of toxicity in a relationship?", "intent": "toxic_relationships"},

    # Relationship satisfaction
    {"problem": "How can we make our relationship better?", "intent": "relationship_satisfaction"},
    {"problem": "What can we do to improve our relationship satisfaction?", "intent": "relationship_satisfaction"},
    {"problem": "How do I feel more satisfied in my relationship?", "intent": "relationship_satisfaction"},
    {"problem": "How can I improve intimacy for a happier relationship?", "intent": "relationship_satisfaction"}

]


# Preprocessing function: Tokenization, Lemmatization, and Wikipedia augmentation
def preprocess_text(text):
    # Tokenize and Lemmatize the input text using spaCy
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    processed_text = " ".join(tokens)

    # Augment with Wikipedia content if available
    wiki_page = wiki_wiki.page(text)  # Use the actual problem statement as the search query
    if wiki_page.exists():
        processed_text += " " + wiki_page.summary[:500]  # Add up to 500 characters from the summary

    return processed_text

# Prepare the dataset
texts = [preprocess_text(item["problem"]) for item in training_data]
labels = [item["intent"] for item in training_data]

# Vectorize the text data using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model using cross-validation
cv_scores = cross_val_score(model, X, labels, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {np.mean(cv_scores)}")

# Save the trained model and vectorizer to disk
with open("relationship_intent_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("relationship_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

# Evaluate on the test set
test_accuracy = model.score(X_test, y_test)
print(f"Test set accuracy: {test_accuracy}")
