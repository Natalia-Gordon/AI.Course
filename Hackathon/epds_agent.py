"""
EPDS (Edinburgh Postnatal Depression Scale) Conversational Agent

סוכן דינמי שמנהל שיחה חכמה:
- יודע מתי לשאול שאלות EPDS
- משתמש ב-NLP לניתוח רגשי
- מתחבר למודל XGBoost להערכת סיכון
- שומר תוצאות בצורה מסודרת
"""

import os
import uuid
import pandas as pd
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from textblob import TextBlob

# Pydantic import (needed for BaseModel)
try:
    from pydantic import BaseModel, Field
except ImportError:
    # Fallback if pydantic not available
    class BaseModel:
        pass
    Field = None

# LangChain imports - handle multiple versions gracefully
LANGCHAIN_AVAILABLE = False
ChatOpenAI = None
BaseTool = None
PromptTemplate = None

# Try to import core LangChain components
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    try:
        from langchain.tools import BaseTool
    except ImportError:
        try:
            from langchain_core.tools import BaseTool
        except ImportError:
            BaseTool = None
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    # LangChain not available - BaseTool will be None
    BaseTool = None  # type: ignore
    print(f"⚠️ LangChain not available. Install: pip install langchain langchain-openai (Error: {e})")

# Try to import agent-related classes (these may not exist in all versions)
# These are imported on-demand in _initialize_langchain() to handle version differences
initialize_agent = None
AgentType = None
create_react_agent = None
AgentExecutor = None
create_agent = None

from dotenv import load_dotenv
load_dotenv()

# EPDS questions (Hebrew)
EPDS_QUESTIONS = [
    "בשבוע האחרון, הצלחתי לצחוק ולראות את הצד המצחיק של דברים",
    "ציפיתי בהנאה לדברים",
    "האשמתי את עצמי ללא סיבה",
    "הרגשתי חרדה או דאגה ללא סיבה",
    "הרגשתי מפוחדת או מבוהלת",
    "הרגשתי שהכול קשה לי מדי",
    "היה לי קשה לישון בגלל דאגות",
    "הרגשתי עצובה או אומללה",
    "הייתי כל כך אומללה שבכיתי",
    "עברו בי מחשבות לפגוע בעצמי"
]

EPDS_COLUMN_NAMES = [
    "צחוק והצד המצחיק",
    "ציפייה בהנאה",
    "האשמה עצמית",
    "דאגה וחרדה",
    "פחד ובהלה",
    "דברים קשים מדי",
    "קושי לישון",
    "עצב ואומללות",
    "בכי",
    "מחשבות פגיעה עצמית"
]

# Expanded list of Hebrew distress keywords and verbal cues
DISTRESS_KEYWORDS = [
    # Direct emotional distress
    "קשה", "עייפה", "בודדה", "לחוצה", "לא מצליחה", "עצובה", "מדוכאת", "ייאוש",
    # Additional emotional states
    "כועסת", "מתוסכלת", "חסרת תקווה", "מפוחדת", "חרדה", "פחד", "בהלה", "דאגה",
    "עצבנית", "מאוכזבת", "אשמה", "אשמה", "אכזבה", "תסכול", "כעס", "זעם",
    # Physical/mental exhaustion
    "תשושה", "מותשת", "חסרת אנרגיה", "לא יכולה", "לא יכולה יותר", "נשברת",
    "לא מצליחה להתמודד", "מוצפת", "מבולבלת", "לא מבינה", "אבודה",
    # Relationship/social distress
    "בודדה", "מבודדת", "לא מבינים אותי", "אף אחד לא מבין", "לא רואה אותי",
    "קושי עם התינוק", "לא מתחברת", "קושי להתחבר", "לא אוהבת", "מפחדת",
    # Self-harm/suicidal ideation (high priority)
    "לא רוצה לחיות", "רוצה למות", "לא כדאי", "למה לי", "אין טעם", "אובדן תקווה",
    "לפגוע בעצמי", "להיפצע", "למות", "סוף", "זה הסוף",
    # Sleep and daily functioning
    "לא ישנה", "לא מצליחה לישון", "נדודי שינה", "עייפה כל הזמן",
    "לא רוצה לקום", "לא רוצה לעשות כלום", "לא מתפקדת",
    # Coping difficulties
    "לא יודעת מה לעשות", "לא יודעת איך להתמודד", "אובדת עצות",
    "חסרת אונים", "חסרת שליטה", "מרגישה לכודה", "אין מוצא"
]


class EPDSState(BaseModel):
    """State for EPDS conversation."""
    session_id: str
    patient_name: str
    epds_answers: List[int] = []
    current_question_index: int = 0
    free_text_collected: bool = False
    free_text: str = ""
    conversation_history: List[Dict[str, str]] = []
    needs_epds_question: bool = True
    needs_free_text: bool = False
    assessment_complete: bool = False


class EPDSAnswerInterpreterTool(BaseTool):
    """Tool for interpreting natural language responses to EPDS questions."""
    
    name: str = "interpret_epds_answer"
    description: str = "Interprets a natural language response to an EPDS question and converts it to a score (0-3). The response should be analyzed based on how often or how much the feeling/behavior occurred in the past week."
    
    def _run(self, question: str, user_response: str) -> str:
        """Interpret natural language response to EPDS question."""
        try:
            response_lower = user_response.lower()
            
            # Direct numeric answer
            numbers = re.findall(r'\d+', user_response)
            if numbers:
                score = int(numbers[0])
                if 0 <= score <= 3:
                    return f"SCORE:{score}"
            
            # Hebrew expressions for frequency/intensity
            # 0 = בכלל לא / מעולם לא
            if any(word in response_lower for word in ['בכלל לא', 'מעולם לא', 'אף פעם לא', 'אפס', '0']):
                return "SCORE:0"
            
            # 1 = לא לעתים קרובות / כמעט אף פעם
            if any(word in response_lower for word in ['לא לעתים קרובות', 'כמעט לא', 'בדרך כלל לא', '1']):
                return "SCORE:1"
            
            # 2 = לפעמים / מדי פעם
            if any(word in response_lower for word in ['לפעמים', 'מדי פעם', 'בינוני', 'קצת', '2']):
                return "SCORE:2"
            
            # 3 = לעתים קרובות מאוד / הרבה / תמיד
            if any(word in response_lower for word in ['לעתים קרובות מאוד', 'הרבה', 'תמיד', 'כמעט תמיד', '3']):
                return "SCORE:3"
            
            # Try sentiment-based interpretation
            blob = TextBlob(user_response)
            sentiment = blob.sentiment.polarity
            
            # If very negative sentiment, likely higher score (more frequent/problematic)
            if sentiment < -0.5:
                return "SCORE:3"
            elif sentiment < -0.2:
                return "SCORE:2"
            elif sentiment < 0:
                return "SCORE:1"
            else:
                return "SCORE:0"
                
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    async def _arun(self, question: str, user_response: str) -> str:
        """Async version."""
        return self._run(question, user_response)


class SentimentAnalysisTool(BaseTool):
    """Tool for analyzing sentiment and distress in text."""
    
    name: str = "analyze_sentiment"
    description: str = "Analyzes text for emotional sentiment and distress keywords. Returns sentiment score (-1 to 1) and detected keywords."
    
    def _run(self, text: str) -> str:
        """Analyze sentiment and detect distress keywords with enhanced sensitivity."""
        try:
            text_lower = text.lower()
            sentiment = TextBlob(text).sentiment.polarity
            
            # Enhanced keyword detection (check both full words and substrings)
            keywords_found = []
            for keyword in DISTRESS_KEYWORDS:
                if keyword.lower() in text_lower:
                    keywords_found.append(keyword)
            
            # Check for high-priority distress indicators (suicidal ideation, self-harm)
            high_priority_keywords = ["לא רוצה לחיות", "רוצה למות", "לפגוע בעצמי", "להיפצע", 
                                     "לא כדאי", "אין טעם", "אובדן תקווה", "זה הסוף"]
            has_high_priority = any(kw in text_lower for kw in high_priority_keywords)
            
            # Enhanced distress level calculation
            # Factor in sentiment, keyword count, and priority indicators
            keyword_count = len(keywords_found)
            if has_high_priority:
                distress_level = "גבוה מאוד"
                urgency = "דחוף"
            elif sentiment < -0.4 or keyword_count >= 3:
                distress_level = "גבוה"
                urgency = "גבוה"
            elif sentiment < -0.2 or keyword_count >= 2:
                distress_level = "בינוני-גבוה"
                urgency = "בינוני"
            elif sentiment < 0 or keyword_count >= 1:
                distress_level = "בינוני"
                urgency = "נמוך-בינוני"
            else:
                distress_level = "נמוך"
                urgency = "נמוך"
            
            result = {
                "sentiment_score": round(sentiment, 2),
                "distress_level": distress_level,
                "urgency": urgency,
                "keywords": keywords_found,
                "high_priority": has_high_priority,
                "keyword_count": keyword_count
            }
            
            return f"ניתוח רגשי: רמת מצוקה {distress_level} (דחיפות: {urgency}), ציון רגשי: {result['sentiment_score']}, מילות מפתח: {', '.join(result['keywords'][:5]) if result['keywords'] else 'אין'}"
        except Exception as e:
            return f"שגיאה בניתוח רגשי: {str(e)}"
    
    async def _arun(self, text: str) -> str:
        """Async version."""
        return self._run(text)


class PPDPredictionTool(BaseTool):
    """Tool for predicting PPD risk using XGBoost model."""
    
    name: str = "predict_ppd_risk"
    description: str = "Predicts PPD risk using the trained XGBoost model. Requires symptom data. Returns risk score and level."
    
    ppd_agent: Optional[Any] = None
    
    def __init__(self, ppd_agent=None, **kwargs):
        super().__init__(**kwargs)
        self.ppd_agent = ppd_agent
    
    def _run(self, age: str = "25-30", feeling_sad: str = "No", irritable: str = "No",
             trouble_sleeping: str = "No", concentration: str = "No", appetite: str = "No",
             feeling_anxious: str = "No", guilt: str = "No", bonding: str = "No",
             suicide_attempt: str = "No") -> str:
        """Predict PPD risk."""
        if self.ppd_agent is None:
            return "מודל PPD לא זמין כרגע"
        
        try:
            result = self.ppd_agent.predict(
                age=age,
                feeling_sad=feeling_sad,
                irritable=irritable,
                trouble_sleeping=trouble_sleeping,
                concentration=concentration,
                appetite=appetite,
                feeling_anxious=feeling_anxious,
                guilt=guilt,
                bonding=bonding,
                suicide_attempt=suicide_attempt
            )
            
            return f"הערכת סיכון PPD: {result['risk_level']} ({result['risk_percentage']}%). {result['explanation'][:200]}"
        except Exception as e:
            return f"שגיאה בהערכת סיכון: {str(e)}"
    
    async def _arun(self, **kwargs) -> str:
        """Async version."""
        return self._run(**kwargs)


def save_epds_assessment(state: EPDSState, sentiment_score: float, keywords: List[str]) -> Tuple[int, int]:
    """Save EPDS assessment to CSV file."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    csv_path = data_dir / "EPDS_answers.csv"
    
    # Get next ID
    if csv_path.exists():
        try:
            existing_df = pd.read_csv(csv_path, encoding='utf-8-sig')
            if 'ID' in existing_df.columns:
                next_id = int(existing_df['ID'].max()) + 1
            else:
                next_id = len(existing_df) + 1
        except Exception:
            next_id = 1
    else:
        next_id = 1
    
    # Create timestamp
    timestamp = datetime.now().strftime("%m/%d/%Y %H:%M")
    
    # Calculate total score
    total_score = sum(state.epds_answers)
    
    # Ensure we have exactly 10 answers
    answers = state.epds_answers.copy()
    while len(answers) < 10:
        answers.append(0)
    
    # Create row data
    row_data = {
        "ID": [next_id],
        "Timestamp": [timestamp],
        "Name": [state.patient_name if state.patient_name else f"Patient_{next_id}"],
        "Total Scores": [total_score]
    }
    
    # Add individual question scores
    for i, col_name in enumerate(EPDS_COLUMN_NAMES):
        row_data[col_name] = [answers[i] if i < len(answers) else 0]
    
    df = pd.DataFrame(row_data)
    
    # Append to existing file or create new one
    if csv_path.exists():
        df.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False, encoding='utf-8-sig')
    
    print(f"✅ Saved EPDS assessment: ID={next_id}, Score={total_score}, Name={state.patient_name}")
    return next_id, total_score


def extract_answer_score(text: str) -> Optional[int]:
    """Extract EPDS answer score (0-3) from text."""
    # Try numeric extraction
    numbers = re.findall(r'\d+', text)
    if numbers:
        score = int(numbers[0])
        if 0 <= score <= 3:
            return score
    
    # Check for Hebrew responses
    text_lower = text.lower()
    if any(word in text_lower for word in ['בכלל לא', 'לא', '0']):
        return 0
    elif any(word in text_lower for word in ['לעתים', '1', 'קצת']):
        return 1
    elif any(word in text_lower for word in ['לפעמים', '2', 'בינוני']):
        return 2
    elif any(word in text_lower for word in ['קרובות', '3', 'הרבה', 'תמיד']):
        return 3
    
    return None


class EPDSAgent:
    """
    סוכן EPDS דינמי עם LangChain
    
    מנהל שיחה חכמה שיודעת:
    - מתי לשאול שאלות EPDS
    - מתי לאסוף טקסט חופשי
    - איך לנתח רגשות
    - איך להתחבר למודל XGBoost
    """
    
    def __init__(self, ppd_agent=None):
        """Initialize EPDS Agent."""
        self.ppd_agent = ppd_agent
        self.llm = None
        self.langchain_agent = None
        self.state: Optional[EPDSState] = None
        
        if LANGCHAIN_AVAILABLE:
            self._initialize_langchain()
    
    def _initialize_langchain(self):
        """Initialize LangChain components."""
        try:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                print("⚠️ OPENAI_API_KEY not found. EPDS agent will work in basic mode.")
                return
            
            # Initialize LLM
            self.llm = ChatOpenAI(
                temperature=0.7,
                model="gpt-4o-mini",
                openai_api_key=openai_api_key
            )
            
            # Create tools
            tools = [
                SentimentAnalysisTool(),
                EPDSAnswerInterpreterTool(),
            ]
            
            # Add PPD prediction tool if agent is available
            if self.ppd_agent is not None:
                tools.append(PPDPredictionTool(ppd_agent=self.ppd_agent))
            
            # Create prompt template for EPDS conversation
            prompt = PromptTemplate.from_template("""אתה סוכן רפואי מקצועי שמנהל הערכה של דיכאון לאחר לידה (EPDS).

המטרה: לנהל שיחה טבעית ולאסוף מידע על מצבה הרגשי של המטופלת.

כלים זמינים:
{tools}

הוראות:
1. התחל בברכה חמה והסבר על התהליך
2. שאל שאלות EPDS אחת אחרי השנייה (0-3), אבל אפשר גם שיחה חופשית
3. אם המטופלת משתפת רגשות, השתמש בכלי analyze_sentiment לניתוח
4. אסוף טקסט חופשי על רגשות בסוף
5. לאחר השלמת EPDS, הצע חיבור למודל XGBoost להערכת סיכון

שאלות EPDS:
{epds_questions}

מצב נוכחי:
- שאלות שנענו: {answered_questions}
- שאלה נוכחית: {current_question_index}
- טקסט חופשי נאסף: {free_text_collected}

תגובה של המטופלת: {user_message}

תגובה:""")
            
            # Try to create agent (various API versions) - import on-demand
            agent_created = False
            
            # Try LangChain 1.x API (create_agent)
            try:
                from langchain.agents import create_agent
                self.langchain_agent = create_agent(
                    model=self.llm,
                    tools=tools,
                    system_prompt="אתה סוכן EPDS מקצועי שמנהל שיחות רגישות עם נשים לאחר לידה."
                )
                agent_created = True
            except (ImportError, AttributeError, Exception) as e:
                # Try create_react_agent (0.3.x) if not created yet
                try:
                    from langchain.agents import create_react_agent, AgentExecutor
                    prompt_template = PromptTemplate.from_template(prompt.template)
                    agent = create_react_agent(self.llm, tools, prompt_template)
                    self.langchain_agent = AgentExecutor(agent=agent, tools=tools, verbose=False)
                    agent_created = True
                except (ImportError, AttributeError, Exception) as e2:
                    # Fallback to initialize_agent (old API) if not created yet
                    try:
                        from langchain.agents import initialize_agent, AgentType
                        self.langchain_agent = initialize_agent(
                            tools=tools,
                            llm=self.llm,
                            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                            verbose=False
                        )
                        agent_created = True
                    except (ImportError, AttributeError, Exception) as e3:
                        # If all fail, we'll continue without LangChain agent
                        # The EPDS agent can still work in basic mode
                        agent_created = False
            
            if not agent_created:
                # Don't raise an error - allow the agent to work without LangChain
                print("⚠️ Could not initialize LangChain agent. EPDS agent will work in basic mode without advanced NLP.")
                self.langchain_agent = None
                return
            
            print("✅ EPDS LangChain agent initialized successfully")
            
        except Exception as e:
            print(f"⚠️ Could not initialize LangChain agent: {e}")
            self.langchain_agent = None
    
    def _convert_question_to_conversational(self, epds_question: str, question_number: int = None) -> str:
        """Convert EPDS question to natural, conversational phrasing using LLM."""
        if self.llm is None:
            # Fallback to simple natural phrasing
            return epds_question.replace("בשבוע האחרון, ", "").replace("הצלחתי", "הצלחת").replace("ציפיתי", "ציפית")
        
        try:
            conversion_prompt = f"""אתה מנהל שיחה טבעית ואמפתית עם אישה לאחר לידה.

השאלה המקורית (EPDS): {epds_question}

תפקידך: להמיר את השאלה לשפה טבעית ושיחה, כאילו אתה שואל חברה או מכרה. 
- הסר את המילה "בשבוע האחרון" אם היא בתחילת השאלה (תוכל להזכיר אותה באופן טבעי)
- שנה את הניסוח לניסוח טבעי ושיחה
- שמור על המשמעות המקורית
- השתמש בשפה חמה ואמפתית
- אל תדבר על "שאלה מספר X" או "שאלה 1" - רק שאל את השאלה באופן טבעי

דוגמה:
מקור: "בשבוע האחרון, הצלחתי לצחוק ולראות את הצד המצחיק של דברים"
שיחה: "אני רוצה לשאול - איך את מרגישה עם הומור וצחוק? את מצליחה לראות את הצד המצחיק של דברים?"

חזור רק עם השאלה המתורגמת, ללא הסברים נוספים."""
            
            conversational_question = self.llm.invoke(conversion_prompt).content.strip()
            return conversational_question
        except Exception:
            # Fallback if LLM fails
            return epds_question.replace("בשבוע האחרון, ", "")
    
    def _get_natural_transition(self, previous_answer_emotional: bool = False) -> str:
        """Get a natural transition phrase between questions."""
        transitions = [
            "בסדר, בואי נמשיך",
            "אני מבינה",
            "תודה ששיתפת",
            "אני מקשיבה",
            "כן, אני מבינה מה את אומרת"
        ]
        
        if previous_answer_emotional:
            emotional_transitions = [
                "תודה על השיתוף הכנה 💙",
                "אני מבינה שזה חשוב לך",
                "תודה על האמון",
                "אני מקשיבה ומבינה"
            ]
            import random
            return random.choice(emotional_transitions)
        else:
            import random
            return random.choice(transitions)
    
    def start_conversation(self, patient_name: str = "") -> str:
        """Start a new EPDS conversation with a natural, human-like greeting."""
        name_part = f" {patient_name}" if patient_name else ""
        self.state = EPDSState(
            session_id=str(uuid.uuid4()),
            patient_name=patient_name if patient_name else f"Patient_{uuid.uuid4().hex[:8]}",
            needs_epds_question=True,
            needs_free_text=False,
            assessment_complete=False
        )
        
        # Natural, conversational greeting - like talking to a friend
        greeting = f"שלום{name_part}! 💙\n\n"
        greeting += f"אני כאן כדי להקשיב ולשוחח איתך על איך את מרגישה בתקופה הזאת.\n\n"
        greeting += f"אני רוצה להכיר אותך קצת יותר ולהבין איך את חווה את התקופה שלאחר הלידה. "
        greeting += f"בואי נשוחח קצת - אין תשובות נכונות או שגויות, פשוט רציתי לשמוע בדיוק איך את מרגישה.\n\n"
        greeting += f"את מוזמנת לענות איך שאת רוצה - במילים שלך, כאוות נפשך. אני מבינה עברית ואקשיב לך בקשב. 💙\n\n"
        
        # Convert first question to conversational format
        first_question_conversational = self._convert_question_to_conversational(EPDS_QUESTIONS[0])
        greeting += f"{first_question_conversational}"
        
        self.state.conversation_history.append({
            "role": "assistant",
            "content": greeting
        })
        
        return greeting
    
    def process_message(self, user_message: str) -> str:
        """Process user message and return agent response."""
        if self.state is None:
            return "אנא התחילי שיחה קודם"
        
        # Add user message to history
        self.state.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Check if we're in EPDS question phase
        if self.state.current_question_index < len(EPDS_QUESTIONS):
            # Get current question
            current_question = EPDS_QUESTIONS[self.state.current_question_index]
            
            # Always check for emotional distress first - even if they gave a numeric answer
            sentiment_tool = SentimentAnalysisTool()
            sentiment_result = sentiment_tool._run(user_message)
            
            # Check for high-priority distress indicators (suicidal ideation, self-harm)
            high_priority_keywords = ["לא רוצה לחיות", "רוצה למות", "לפגוע בעצמי", "להיפצע", 
                                     "לא כדאי", "אין טעם", "אובדן תקווה", "זה הסוף"]
            text_lower = user_message.lower()
            has_high_priority_distress = any(kw in text_lower for kw in high_priority_keywords)
            
            # If high priority distress detected, respond with immediate support
            if has_high_priority_distress:
                response = "💙 אני מבינה שאת חווה קושי גדול. חשוב לי שתדעי שאת לא לבד.\n\n"
                response += "אם את חווה מחשבות קשות או מחשבות על פגיעה בעצמך, אני ממליצה בחום לפנות מיד לעזרה מקצועית:\n"
                response += "• ער״ן (חירום נפשי): 1201\n"
                response += "• נט״ל: 1-800-363-363\n"
                response += "• או פני לחדר מיון קרוב\n\n"
                response += "אני כאן להקשיב. רוצה לשתף עוד?\n\n"
                # Continue naturally with conversational question
                current_q_conversational = self._convert_question_to_conversational(current_question)
                response += f"אם תרצי, בואי נמשיך. {current_q_conversational}"
            
            # Try to extract numeric answer
            answer = extract_answer_score(user_message)
            
            if answer is not None:
                # Valid answer - save and move to next question
                self.state.epds_answers.append(answer)
                self.state.current_question_index += 1
                
                # Check for emotional content even in numeric answers
                has_distress_keywords = any(kw in user_message for kw in DISTRESS_KEYWORDS)
                
                if self.state.current_question_index < len(EPDS_QUESTIONS):
                    # More questions to ask - use natural, conversational phrasing
                    next_epds_q = EPDS_QUESTIONS[self.state.current_question_index]
                    next_q_conversational = self._convert_question_to_conversational(next_epds_q, self.state.current_question_index + 1)
                    
                    if has_distress_keywords and not has_high_priority_distress:
                        # Acknowledge the emotional sharing before continuing naturally
                        transition = self._get_natural_transition(previous_answer_emotional=True)
                        response = f"{transition}. {next_q_conversational}"
                    else:
                        # Natural transition to next question
                        transition = self._get_natural_transition(previous_answer_emotional=False)
                        response = f"{transition}. {next_q_conversational}"
                else:
                    # All questions answered - ask for free text in a sensitive way
                    self.state.needs_free_text = True
                    response = "תודה רבה על השיתוף הכנה והאמון 💙\n\n"
                    response += "אם תרצי, אני כאן להקשיב - רוצה לשתף במשפט או שניים איך את מרגישה רגשית בתקופה הזאת? "
                    response += "זה יעזור לי להבין טוב יותר את המצב שלך, אבל זה לגמרי אופציונלי."
            else:
                # No clear answer extracted - use LLM to understand and respond naturally
                has_distress_keywords = any(kw in user_message for kw in DISTRESS_KEYWORDS)
                
                if self.llm is not None and not has_high_priority_distress:
                    # Use LLM to generate a natural, empathetic response
                    try:
                        llm_context = f"""אתה סוכן רפואי אמפתי שמנהל שיחה עם אישה לאחר לידה.

השאלה הנוכחית: {current_question}

התשובה של המטופלת: {user_message}

התשובה לא ברורה כציון מספרי (0-3). תפקידך:
1. להבין את התשובה במילים הטבעיות שלה
2. לאשר שאת מבין/ה (אמפתיה)
3. לנסות לפרש את התשובה לציון 0-3 אם אפשר
4. אם לא אפשר, להזמין אותה לפרט קצת יותר

חזור עם תגובה קצרה, אמפתית וטבעית בעברית. אם הצלחת לפרש לציון, ציין אותו בסוף בצורה עדינה."""
                        
                        llm_response = self.llm.invoke(llm_context).content.strip()
                        response = llm_response
                        
                        # Try to extract any score the LLM might have inferred
                        extracted_score = extract_answer_score(llm_response)
                        if extracted_score is not None:
                            answer = extracted_score
                            self.state.epds_answers.append(answer)
                            self.state.current_question_index += 1
                            if self.state.current_question_index < len(EPDS_QUESTIONS):
                                # Continue naturally to next question
                                next_epds_q = EPDS_QUESTIONS[self.state.current_question_index]
                                next_q_conversational = self._convert_question_to_conversational(next_epds_q, self.state.current_question_index + 1)
                                transition = self._get_natural_transition(previous_answer_emotional=True)
                                response += f"\n\n{transition}. {next_q_conversational}"
                            else:
                                self.state.needs_free_text = True
                                response += "\n\nתודה רבה על השיתוף הכנה 💙\n"
                                response += "אם תרצי, אני כאן להקשיב - רוצה לשתף במשפט או שניים איך את מרגישה רגשית בתקופה הזאת?"
                        else:
                            # Add the current question again in conversational format
                            current_q_conversational = self._convert_question_to_conversational(current_question)
                            response += f"\n\n{current_q_conversational}"
                    except Exception as e:
                        # Fallback if LLM fails - use conversational format
                        current_q_conversational = self._convert_question_to_conversational(current_question)
                        if has_distress_keywords and not has_high_priority_distress:
                            transition = self._get_natural_transition(previous_answer_emotional=True)
                            response = f"{transition}. {current_q_conversational}"
                        else:
                            transition = self._get_natural_transition(previous_answer_emotional=False)
                            response = f"{transition}. {current_q_conversational}"
                else:
                    # No LLM available - use conversational format with rule-based response
                    current_q_conversational = self._convert_question_to_conversational(current_question)
                    if has_distress_keywords and not has_high_priority_distress:
                        transition = self._get_natural_transition(previous_answer_emotional=True)
                        response = f"{transition}. {current_q_conversational}"
                    else:
                        transition = self._get_natural_transition(previous_answer_emotional=False)
                        response = f"{transition}. {current_q_conversational}"
        
        elif self.state.needs_free_text and not self.state.free_text_collected:
            # Collecting free text - enhanced emotional sensitivity
            self.state.free_text = user_message
            self.state.free_text_collected = True
            
            # Analyze sentiment with enhanced detection
            sentiment, keywords = self._analyze_sentiment(user_message)
            
            # Check for high-priority distress
            text_lower = user_message.lower()
            high_priority_keywords = ["לא רוצה לחיות", "רוצה למות", "לפגוע בעצמי", "להיפצע", 
                                     "לא כדאי", "אין טעם", "אובדן תקווה", "זה הסוף"]
            has_high_priority = any(kw in text_lower for kw in high_priority_keywords)
            
            # Save assessment
            record_id, total_score = save_epds_assessment(
                self.state,
                sentiment,
                keywords
            )
            
            # Determine risk level
            risk_assessment = self._assess_risk(total_score, sentiment, keywords)
            
            # Generate sensitive, supportive response
            response = f"תודה רבה על השיתוף הכנה והאמון 💙\n\n"
            
            # If high priority distress detected, add immediate support message
            if has_high_priority:
                response += "⚠️ אני רוצה להדגיש: אם את חווה מחשבות קשות או מחשבות על פגיעה בעצמך, "
                response += "אני ממליצה בחום לפנות מיד לעזרה מקצועית:\n"
                response += "• ער״ן (חירום נפשי): 1201\n"
                response += "• נט״ל: 1-800-363-363\n"
                response += "• או פני לחדר מיון קרוב\n\n"
            
            response += f"📊 תוצאות ההערכה:\n"
            response += f"ציון EPDS: {total_score}/30\n"
            response += f"רמת סיכון: {risk_assessment['risk_level']}\n\n"
            response += f"💙 {risk_assessment['recommendation']}\n\n"
            
            # Add supportive message based on risk level
            if total_score >= 13:
                response += "אני רואה שאת חווה קושי משמעותי. זה לגמרי נורמלי ונפוץ, ואת לא לבד. "
                response += "הרבה אימהות חוות תחושות דומות לאחר לידה. מומלץ מאוד לשקול פניה לייעוץ מקצועי שיכול לעזור. "
                response += "יש תמיכה זמינה, ואת ראויה לקבל אותה. 💙\n\n"
            elif total_score >= 10:
                response += "אני רואה שיש תחושות של קושי. חשוב לעקוב אחרי המצב ולהיות קשובה לעצמך. "
                response += "אם התחושות נמשכות או מתחזקות, זה בסדר לבקש עזרה. 💙\n\n"
            else:
                response += "תודה על השיתוף. אם תרגישי שמשהו משתנה או אם תחושי צורך, "
                response += "תמיד אפשר לשוב ולשוחח או לפנות לעזרה. 💙\n\n"
            
            if self.ppd_agent is not None:
                response += f"💡 אם תרצי, אני יכולה לעזור להעריך את המצב גם עם כלי נוסף. "
                response += f"זה אופציונלי לחלוטין - רק אם את מרגישה בנוח."
            
            response += f"\n✅ התשובות נשמרו בהצלחה (רשומה #{record_id})"
            
            self.state.assessment_complete = True
        
        else:
            # Conversation completed or other state
            if self.state.assessment_complete:
                # Try to extract symptom info for XGBoost prediction
                if self.ppd_agent is not None and any(word in user_message.lower() for word in ['כן', 'סמפטומים', 'תסמינים', 'סימפטומים']):
                    # User wants XGBoost prediction - would need to extract symptoms from conversation
                    response = "אני יכול לעזור עם הערכת XGBoost. כדי לקבל הערכה מדויקת, אני צריך מידע על:\n"
                    response += "- גיל\n- תחושות של עצב\n- עצבנות כלפי התינוק/בן הזוג\n- בעיות שינה\n- קשיי ריכוז\n- שינויים בתיאבון\n- חרדה\n- רגשות אשמה\n- קשיי קשר עם התינוק\n- מחשבות אובדניות\n\nאם תרצי, אני יכול לעזור למלא את השאלון המלא."
                else:
                    response = "הערכה הושלמה. יש משהו נוסף שברצונך לדון בו?"
            else:
                response = "אנא לחצי על 'התחל הערכה' כדי להתחיל"
        
        # Add response to history
        self.state.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def _analyze_sentiment(self, text: str) -> Tuple[float, List[str]]:
        """Analyze sentiment and detect keywords."""
        try:
            sentiment = TextBlob(text).sentiment.polarity
            keywords = [k for k in DISTRESS_KEYWORDS if k in text]
            return sentiment, keywords
        except:
            return 0.0, []
    
    def _assess_risk(self, epds_score: int, sentiment: float, keywords: List[str]) -> Dict[str, str]:
        """Assess risk level based on EPDS score, sentiment, and keywords."""
        # EPDS risk levels
        if epds_score >= 13:
            risk_level = "גבוה"
            recommendation = "מומלץ מאוד לפנות לייעוץ מקצועי בהקדם"
        elif epds_score >= 10:
            risk_level = "בינוני-גבוה"
            recommendation = "מומלץ לעקוב אחרי המצב ולשקול ייעוץ מקצועי"
        else:
            risk_level = "נמוך-בינוני"
            recommendation = "מומלץ להמשיך לעקוב אחרי המצב"
        
        # Adjust based on sentiment and keywords
        if sentiment < -0.3 or len(keywords) >= 2:
            if risk_level == "נמוך-בינוני":
                risk_level = "בינוני"
                recommendation = "מומלץ לשקול שיחה עם איש מקצוע"
        
        return {
            "risk_level": risk_level,
            "recommendation": recommendation
        }
    
    def reset(self):
        """Reset conversation state."""
        self.state = None

