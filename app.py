import gradio as gr
import random, json, os, re
from gtts import gTTS
import openai

# ---------- CONFIG ----------
openai.api_key = os.getenv("MY_API_KEY")  # Set in Hugging Face Secrets
openai.api_base = "https://api.groq.com/openai/v1"
MODEL_NAME = "llama-3.1-8b-instant"
PROGRESS_FILE = "progress.json"

# ---------- FIXED DUAS LIST ----------
DUAS = [
    {"occasion": "BEFORE EATING", "arabic": "بِسْمِ الله", "english": "In the name of Allah"},
    {"occasion": "AFTER EATING", "arabic": "الْحَمْدُ لِلَّهِ", "english": "All praise is for Allah"},
    {"occasion": "BEFORE SLEEPING", "arabic": "بِاسْمِكَ اللَّهُمَّ أَمُوتُ وَأَحْيَا", "english": "In Your name O Allah, I die and I live"},
    {"occasion": "AFTER WAKING UP", "arabic": "الْحَمْدُ لِلَّهِ الَّذِي أَحْيَانَا بَعْدَ مَا أَمَاتَنَا وَإِلَيْهِ النُّشُورُ", "english": "All praise is for Allah, who gave us life after death, and to Him is the return"},
    {"occasion": "BEFORE ENTERING TOILET", "arabic": "اللَّهُمَّ إِنِّي أَعُوذُ بِكَ مِنَ الْخُبُثِ وَالْخَبَائِثِ", "english": "O Allah, I seek refuge in You from male and female devils"},
    {"occasion": "AFTER LEAVING TOILET", "arabic": "غُفْرَانَكَ", "english": "I ask You (Allah) for Your forgiveness"},
    {"occasion": "ENTERING HOME", "arabic": "بِسْمِ اللهِ وَلَجْنَا وَبِسْمِ اللهِ خَرَجْنَا وَعَلَى اللهِ رَبِّنَا تَوَكَّلْنَا", "english": "In the name of Allah we enter, in the name of Allah we leave, and upon our Lord we rely"},
    {"occasion": "LEAVING HOME", "arabic": "بِسْمِ اللهِ تَوَكَّلْتُ عَلَى الله", "english": "In the name of Allah, I place my trust in Allah"},
    {"occasion": "WHEN SNEEZING", "arabic": "الْحَمْدُ لِلَّهِ", "english": "All praise is for Allah"},
    {"occasion": "HEARING SOMEONE SNEEZE", "arabic": "يَرْحَمُكَ الله", "english": "May Allah have mercy on you"},
    {"occasion": "WHEN WEARING NEW CLOTHES", "arabic": "الْحَمْدُ لِلَّهِ الَّذِي كَسَانِي هَذَا", "english": "All praise is for Allah who clothed me with this"},
    {"occasion": "WHEN SEEING SOMETHING BEAUTIFUL", "arabic": "سُبْحَانَ الله", "english": "Glory be to Allah"},
    {"occasion": "WHEN STARTING TO STUDY", "arabic": "رَبِّ زِدْنِي عِلْمًا", "english": "My Lord, increase me in knowledge"},
    {"occasion": "WHEN TRAVELING", "arabic": "سُبْحَانَ الَّذِي سَخَّرَ لَنَا هَذَا وَمَا كُنَّا لَهُ مُقْرِنِينَ", "english": "Glory to Him who has subjected this (transport) to us, and we could never have done it ourselves"},
]

# ---------- PROGRESS TRACKER ----------
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {
        "points": 0,
        "duas_learned": 0,
        "stories_listened": 0,
        "quizzes_attempted": 0,
        "duas_list": []
    }

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)

def add_points(points):
    progress = load_progress()
    progress["points"] += points
    save_progress(progress)

# ---------- AI HELPER ----------
def ask_groq(prompt, max_tokens=400):
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a friendly Islamic teacher for children age 5-12. When telling stories about Prophets or Companions, only use authentic sources: the Qur’an and Sahih Hadith (Bukhari, Muslim, Abu Dawood, Tirmidhi, Nasa’i, Ibn Majah) and trusted tafsir like Ibn Kathir.  Always give the reference (Surah/Ayah or Hadith book + number).Do not mix Biblical or cultural stories. For general moral stories (not about Prophets/Companions), you may create simple fictional examples."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"AI Error: {str(e)}"

# ---------- STORY GENERATOR ----------
def generate_story(topic):
    prompt = f"""
    Write one Islamic story for children (age 5-12) about {topic}.
    Rules:
    - If the story involves any Prophet or Companion, ONLY use authentic sources (Qur’an, Sahih Hadith, reliable Tafsir).
    - Always give the reference (Surah name + ayah number OR Hadith collection + number).
    - If the story detail is not in authentic sources, clearly say: 'This is not mentioned in authentic sources.'
    - Do NOT invent or add extra details.
    - For topics not about Prophets/Companions, you may create a fictional moral story, but it must align with Islamic values.
    - Length: 300–500 words
    - Style: Simple, engaging, age-appropriate
    - End with a clear moral
    - Only 1 story, not multiple
    """
    story = ask_groq(prompt, max_tokens=700)

    add_points(5)
    progress = load_progress()
    progress["stories_listened"] += 1
    save_progress(progress)

    audio_file = "story.mp3"
    try:
        tts = gTTS(story, lang='en')
        tts.save(audio_file)
    except:
        audio_file = None

    return story, audio_file

# ---------- FIXED DUA GENERATOR ----------
def generate_dua():
    progress = load_progress()
    learned_duas = progress.get("duas_list", [])

    available_duas = [d for d in DUAS if d["occasion"] not in [ld.get("occasion") for ld in learned_duas]]
    if not available_duas:
        available_duas = DUAS  # reset if all learned

    dua = random.choice(available_duas)
    progress["duas_learned"] += 1
    progress["duas_list"].append(dua)
    add_points(2)
    save_progress(progress)

    return f"Let's learn the dua we should say {dua['occasion']}:\n\nArabic: {dua['arabic']}\nEnglish: {dua['english']}"

# ---------- QUIZ GENERATOR ----------
def generate_quiz():
    prompt = """Create one multiple-choice Islamic quiz question for children (age 5-12).
Format exactly:
Question: <text>
A) <option>
B) <option>
C) <option>
D) <option>
Answer: <correct option letter>"""
    
    quiz_text = ask_groq(prompt, max_tokens=200)
    question_match = re.search(r"Question:(.*)", quiz_text)
    options = re.findall(r"[A-D]\)(.*)", quiz_text)
    answer_match = re.search(r"Answer:\s*([A-D])", quiz_text)

    options = [opt.strip() for opt in options[:4]]
    if len(options) < 4:
        options += [f"Option {i}" for i in range(len(options)+1, 5)]

    if question_match and options and answer_match:
        question = question_match.group(1).strip()
        correct_index = "ABCD".index(answer_match.group(1).upper())
        correct_answer = options[correct_index]
        return question, gr.update(choices=options, value=None), correct_answer
    else:
        return "AI failed to generate a quiz.", gr.update(choices=["Option 1","Option 2","Option 3","Option 4"], value=None), "Option 1"

def check_quiz_answer(user_answer, correct_answer):
    if not user_answer:
        return "⚠️ Please select an answer!"
    progress = load_progress()
    progress["quizzes_attempted"] += 1
    save_progress(progress)

    if user_answer.strip().lower() == correct_answer.strip().lower():
        add_points(5)
        return "✅ Correct! 🌟"
    else:
        return f"❌ Oops! The correct answer was: {correct_answer}"

# ---------- PROGRESS ----------
def my_progress():
    progress = load_progress()
    last_5_duas = progress.get("duas_list", [])[-5:]
    duas_display = "\n".join([f"- {d['occasion']}: {d['arabic']} ({d['english']})" for d in last_5_duas]) or "No duas learned yet."
    return (f"Points: {progress['points']}\n"
            f"Duas Learned: {progress['duas_learned']}\n"
            f"Stories Listened: {progress['stories_listened']}\n"
            f"Quizzes Attempted: {progress['quizzes_attempted']}\n\n"
            f"Last 5 Duas Learned:\n{duas_display}")

# ---------- GRADIO UI ----------
with gr.Blocks() as demo:
    gr.Markdown("# 🌙 Little Muslim Companion\n_As-salamu Alaikum! Let's learn and have fun together!_")

    with gr.Tab("📖 Storytime"):
        story_input = gr.Textbox(label="Enter topic for story (e.g., honesty, kindness)")
        story_output = gr.Textbox(label="Story")
        story_audio = gr.Audio(label="Story Voice", type="filepath")
        story_btn = gr.Button("Generate Story")
        story_btn.click(fn=generate_story, inputs=story_input, outputs=[story_output, story_audio])

    with gr.Tab("🙏 Daily Duas"):
        dua_output = gr.Textbox(label="Your Dua")
        dua_btn = gr.Button("Get a Dua")
        dua_btn.click(fn=generate_dua, inputs=None, outputs=dua_output)

    with gr.Tab("🎮 Quizzes"):
        quiz_question = gr.Textbox(label="Quiz Question")
        quiz_options = gr.Radio(label="Options", choices=[])
        quiz_result = gr.Textbox(label="Result")
        quiz_answer = gr.State("")
        quiz_btn = gr.Button("Generate Quiz")
        submit_btn = gr.Button("Submit Answer")

        quiz_btn.click(fn=generate_quiz, inputs=None, outputs=[quiz_question, quiz_options, quiz_answer])
        submit_btn.click(fn=check_quiz_answer, inputs=[quiz_options, quiz_answer], outputs=quiz_result)

    with gr.Tab("⭐ My Progress"):
        progress_output = gr.Textbox(label="Progress")
        progress_btn = gr.Button("Refresh Progress")
        progress_btn.click(fn=my_progress, inputs=None, outputs=progress_output)

demo.launch()
