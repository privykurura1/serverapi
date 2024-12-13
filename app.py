from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

# Load the trained model and vectorizer
with open('chatbot_model.pkl', 'rb') as model_file:
    vectorizer, classifier = pickle.load(model_file)

app = Flask(__name__)

# Enable CORS to allow cross-origin requests
CORS(app)

@app.route('/')
def home():
    return "Welcome to the Relationship Chatbot! Ask me a question."

@app.route('/ask', methods=['POST'])
def ask():
    # Get user input from the request
    user_input = request.json.get('sentence')  # Ensure Flutter sends 'sentence'

    if not user_input:
        return jsonify({"error": "No question provided"}), 400

    # Vectorize the user's question
    user_input_vec = vectorizer.transform([user_input])

    # Predict the intent
    predicted_intent = classifier.predict(user_input_vec)[0]

    # Map intents to responses (simplified)
    responses = {
        "ask_love_signs": "If she texts you back, she might be interested.",
        "ask_lost_love": "If someone doesn't show interest or affection anymore, they might have moved on.",
        "social_media_ex": "It's important to set boundaries and maintain respect on social media with your ex.",
        "heartbreak": "It’s okay to feel heartbroken. Give yourself time to heal.",
        "thank_you": "You're welcome! I'm always here to help.",
        "greet": "Hello! How can I assist you today?",
        "communication": "Try being open, clear, and listen actively to communicate effectively.",
        "conflict_resolution": "Compromise, listen to each other, and work as a team to resolve conflicts.",
        "active_listening": "Active listening means being fully present and understanding your partner’s point of view.",
        "trust_rebuilding": "Rebuilding trust takes time, patience, and consistent actions.",
        "jealousy": "Trust and communication are key to overcoming jealousy in a relationship.",
        'honesty': 'Honesty is the foundation of trust. Be open about your thoughts and feelings, and encourage your partner to do the same.',
        'rekindling_intimacy': 'To rekindle intimacy, prioritize spending quality time together, express affection regularly, and explore each other’s needs and desires.',
        'libido_differences': 'When sex drives differ, open communication is essential. Discuss your feelings, desires, and expectations, and find ways to compromise and understand each other’s needs.',
        'intimacy_fears': 'Overcoming intimacy fears involves building trust, communicating openly, and taking things slowly. Seek professional support if needed.',
        'independence_togetherness': 'Balancing independence and togetherness is about respecting each other’s personal space while also nurturing your connection and shared activities.',
        'boundary_setting': 'Healthy boundaries are crucial. Discuss your needs and expectations clearly, and respect each other’s boundaries to maintain mutual respect and harmony.',
        'controlling_behavior': 'Dealing with a controlling partner requires clear communication about your boundaries. Stand firm, be assertive, and consider seeking support if the situation escalates.',
        'long_distance_intimacy': 'Maintaining intimacy in a long-distance relationship requires regular communication, emotional support, and creative ways to stay connected despite the distance.',
        'long_distance_connection': 'To avoid drifting apart in a long-distance relationship, prioritize regular check-ins, stay emotionally engaged, and make time for shared experiences, even from afar.',
        'long_distance_trust': 'Trust is vital in long-distance relationships. Stay transparent, communicate openly, and show commitment through actions to strengthen your trust.',
        'relationship_health': 'A healthy relationship is built on trust, communication, mutual respect, and shared goals. Regularly check in with each other about your needs and feelings.',
        'toxic_relationships': 'A toxic relationship is characterized by manipulation, abuse, or constant negativity. It’s important to set boundaries, seek help, and consider ending the relationship if it’s unhealthy.',
        'relationship_satisfaction': 'Improving relationship satisfaction involves open communication, meeting each other’s needs, and working together towards common goals.',
        'family_dynamics': 'Balancing your relationship with your family requires setting clear boundaries and ensuring mutual respect between your partner and family members.',
        'family_interference': 'When dealing with a controlling or overbearing family member, it’s important to set firm boundaries and prioritize your relationship’s well-being.',
        'financial_management': 'Managing finances as a couple requires transparency, open communication, and shared financial goals. Consider discussing budgeting, saving, and financial priorities.',
        'money_discussions': 'Talking about money without arguing involves staying calm, avoiding blame, and focusing on shared financial goals. Practice empathy and understanding.',
        'work_life_balance': 'Balancing work and relationship requires clear priorities, time management, and communication. Ensure you’re giving enough time to nurture your connection outside of work.',
        'career_support': 'Supporting each other’s career goals involves mutual encouragement, flexibility, and understanding each other’s professional aspirations and challenges.',
        'cultural_differences': 'Navigating cultural differences requires empathy, respect, and willingness to learn from each other’s perspectives. Embrace diversity and seek understanding.',
        'religious_differences': 'Religious differences can be challenging, but mutual respect and understanding are essential. Open, non-judgmental conversations can help bridge the gap.',
        'age_gap_relationship': 'In age-gap relationships, open communication and understanding each other’s needs, experiences, and challenges are essential to navigate differences.',
        'life_stage_differences': 'When dealing with different life stages, it’s important to understand and respect each other’s experiences and needs while maintaining open communication.',
        'infidelity_recovery': 'Recovering from infidelity involves rebuilding trust, offering sincere apologies, and committing to transparency. Seek therapy or counseling to aid healing.',
        'trust_rebuilding_after_infidelity': 'Trust can be rebuilt after infidelity with patience, consistent effort, open communication, and sincere apologies from both parties.',
        'domestic_violence_escape': 'If you’re in an abusive relationship, it’s essential to seek safety immediately. Contact a local support organization or a trusted person for help.',
        'domestic_violence_help': 'There are many resources available to help victims of domestic violence, including hotlines, shelters, and support groups. Please reach out for help.',
        'online_relationship_health': 'Maintaining a healthy online relationship requires trust, clear communication, and respecting privacy. Avoid secrets and be honest about your feelings.',
        'online_relationship_safety': 'Protect yourself in an online relationship by keeping your personal information secure, meeting in safe places, and trusting your instincts if something feels off.',
        'sexual_satisfaction': 'Improving sexual satisfaction requires open communication, mutual understanding, and experimenting to understand each other’s preferences and boundaries.',
        'sexual_communication': 'Effective sexual communication involves discussing desires, boundaries, and any concerns openly with your partner to ensure mutual understanding and comfort.',
        'tech_addiction': 'Balancing online and offline relationships requires setting boundaries with technology and prioritizing face-to-face interactions to nurture real-world connections.',
        'social_media_addiction': 'If your partner’s social media use is excessive, discuss your concerns openly and set healthy boundaries for time spent online.',
        'online_privacy': 'Protecting privacy in an online relationship involves ensuring that both partners respect each other’s personal space, passwords, and online presence.',
        'online_safety': 'To avoid online scams and fraud, stay alert to suspicious behavior, avoid sharing personal information with strangers, and report any concerning activities.',
        'generational_gap': 'Bridging a generational gap requires mutual respect, patience, and an openness to learn from each other’s experiences and viewpoints.',
        'communication_styles': 'Handling different communication styles requires patience, understanding, and compromise. Be willing to adapt your approach for clearer communication.',
        'shared_interests': 'Finding common ground involves sharing activities that both of you enjoy. Focus on mutual hobbies, experiences, and interests to strengthen your bond.',
        'health_support': 'Supporting each other through a health crisis involves emotional care, encouragement, and practical help. Stay connected and show your partner that you’re there for them.',
        'emotional_support': 'Providing emotional support means being present, listening actively, and offering comfort and reassurance during difficult times.',
        'intimacy_and_health': 'Maintaining intimacy during illness or disability requires patience, understanding, and adapting to new physical and emotional needs.',
        'addiction_support': 'Supporting a partner through addiction requires empathy, understanding, and encouraging professional treatment while protecting your own well-being.',
        'self_protection': 'Protecting yourself from the negative effects of a partner’s addiction involves setting boundaries, seeking support, and maintaining your own emotional health.',
        'long_distance_relationships': 'Navigating long-distance relationships requires trust, communication, and making time for emotional connection despite the physical distance.',
        'parenthood_challenges': 'Adjusting to parenthood requires patience, teamwork, and flexibility. Support each other in the transition and find a balance that works for both of you.',
        'aging_and_retirement': 'Navigating aging and retirement requires open communication about future goals, health care plans, and ways to enjoy life together in the next stage.',
        'personal_growth': 'Supporting each other’s personal growth involves encouragement, flexibility, and creating space for each partner’s individual aspirations.',
        'individuality_and_togetherness': 'Balancing individuality and togetherness requires respecting each other’s personal needs while nurturing your shared life together.',
        'avoiding_growing_apart': 'To avoid growing apart, prioritize quality time, emotional connection, and open communication about your relationship’s needs.',
        'financial_stress': 'Managing financial stress as a couple requires open dialogue about finances, setting a budget, and working together toward shared financial goals.',
        'job_insecurity': 'Dealing with job insecurity requires open communication, emotional support, and making adjustments to manage stress and anxiety together.',
        'social_and_political_issues': 'Discussing social and political issues requires respect for differing opinions, understanding, and finding common ground.',
        'mental_health_support': 'Support your partner through mental health struggles by offering emotional support, encouraging professional help, and fostering open communication.',
        'self_care_and_relationships': 'Taking care of your own mental health is crucial in a relationship. Make time for self-care, set boundaries, and prioritize your well-being.',
        'sexual_health': 'Addressing sexual dysfunction involves open communication, understanding each other’s needs, and seeking professional help if necessary.',
        'midlife_crisis': 'Navigating a midlife crisis involves understanding the changes your partner is going through, offering emotional support, and adapting to new realities together.',
        'grief_and_loss': 'Coping with grief requires mutual support, empathy, and patience. Allow each other to grieve in your own way while being there for each other.',
        'generational_conflict': 'Resolving generational conflicts involves understanding differing perspectives, acknowledging cultural differences, and maintaining respect despite disagreements.',
        'family_conflict': 'To resolve conflicts with in-laws or extended family, establish clear boundaries, communicate openly, and work together as a team to find solutions.',
        'sustainability_and_relationships': 'Balancing sustainability and relationships involves discussing values, making conscious choices, and supporting each other in adopting eco-friendly habits.',
        'environment_and_relationships': 'Discussing environmental issues requires a collaborative approach, respect for differing opinions, and finding actionable ways to make a positive impact together.',
        'domestic_abuse_escape': 'If you’re in an abusive relationship, prioritize your safety. Reach out to local shelters or support organizations to escape safely.',
        'domestic_abuse_help': 'There are various organizations and helplines that can assist you in escaping an abusive situation. Please seek help immediately.',
        'online_harassment': 'Protect yourself from online harassment by blocking abusive users, reporting incidents, and maintaining strong privacy settings on your accounts.',
        'cyberbullying_support': 'If your partner is experiencing cyberbullying, offer support, help them block the aggressors, and report the incidents to the platform or authorities.',
        'good_girlfriend': 'To attract a good girlfriend, focus on being yourself, developing emotional maturity, and showing genuine interest in her feelings and needs. Build a strong foundation of trust and respect in any relationship.',
        'overprotective_boyfriend': 'Signs of an overprotective boyfriend may include him wanting to control your actions, being overly possessive, or limiting your social interactions. It is important to have open communication to address these concerns and set healthy boundaries.',
        'not_loved': 'If your partner is withdrawing emotionally, avoiding communication, or not showing affection, these could be signs that they are no longer as invested. However, it is crucial to talk openly with your partner to understand what might be going on.',
        'get_her_back': 'If you want to win her back, focus on understanding what went wrong in the relationship, showing genuine remorse, and making meaningful changes. It’s important to respect her space and give her time to make a decision.',
        'fall_in_love': 'Love is not something you can force or control. It develops naturally over time through emotional connection, mutual respect, and shared experiences. Focus on being a good listener, showing kindness, and building a genuine bond with her.'
    }

    # Return the response based on the predicted intent
    response = responses.get(predicted_intent, "I'm not sure how to help with that, but I'm here for you!")
    return jsonify({"response": response})

if __name__ == '__main__':
    # Make the Flask app accessible to other devices in the local network
    app.run(debug=True, host='0.0.0.0', port=5000)