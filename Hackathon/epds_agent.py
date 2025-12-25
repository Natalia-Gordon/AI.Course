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

DISTRESS_KEYWORDS = ["קשה", "עייפה", "בודדה", "לחוצה", "לא מצליחה", "עצובה", "מדוכאת", "ייאוש"]


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


class SentimentAnalysisTool(BaseTool):
    """Tool for analyzing sentiment and distress in text."""
    
    name: str = "analyze_sentiment"
    description: str = "Analyzes text for emotional sentiment and distress keywords. Returns sentiment score (-1 to 1) and detected keywords."
    
    def _run(self, text: str) -> str:
        """Analyze sentiment and detect distress keywords."""
        try:
            sentiment = TextBlob(text).sentiment.polarity
            keywords_found = [k for k in DISTRESS_KEYWORDS if k in text]
            
            # Determine distress level
            if sentiment < -0.3 or len(keywords_found) >= 2:
                distress_level = "גבוה"
            elif sentiment < 0 or len(keywords_found) >= 1:
                distress_level = "בינוני"
            else:
                distress_level = "נמוך"
            
            result = {
                "sentiment_score": round(sentiment, 2),
                "distress_level": distress_level,
                "keywords": keywords_found,
                "interpretation": f"רמת מצוקה: {distress_level}"
            }
            
            return f"ניתוח רגשי: {result['interpretation']}, ציון: {result['sentiment_score']}, מילות מפתח: {', '.join(result['keywords']) if result['keywords'] else 'אין'}"
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
    
    def start_conversation(self, patient_name: str = "") -> str:
        """Start a new EPDS conversation."""
        self.state = EPDSState(
            session_id=str(uuid.uuid4()),
            patient_name=patient_name if patient_name else f"Patient_{uuid.uuid4().hex[:8]}",
            needs_epds_question=True,
            needs_free_text=False,
            assessment_complete=False
        )
        
        greeting = f"שלום! אני כאן כדי לעזור לך להעריך את המצב הרגשי שלך לאחר הלידה. 😊\n\n"
        greeting += f"אני אמנחה אותך דרך שאלות קצרות. תוכלי לענות עם מספר 0-3 או במילים:\n"
        greeting += f"0 = בכלל לא\n1 = לא לעתים קרובות\n2 = לפעמים\n3 = לעתים קרובות מאוד\n\n"
        greeting += f"אם תרצי לשתף רגשות נוספים, תמיד אפשר! 💙\n\n"
        greeting += f"בואי נתחיל - שאלה 1:\n{EPDS_QUESTIONS[0]}"
        
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
            # Try to extract answer
            answer = extract_answer_score(user_message)
            
            if answer is not None:
                # Valid answer - save and move to next question
                self.state.epds_answers.append(answer)
                self.state.current_question_index += 1
                
                if self.state.current_question_index < len(EPDS_QUESTIONS):
                    # More questions to ask
                    next_q = EPDS_QUESTIONS[self.state.current_question_index]
                    response = f"תודה! שאלה {self.state.current_question_index + 1}:\n{next_q}"
                else:
                    # All questions answered - ask for free text
                    self.state.needs_free_text = True
                    response = "תודה על התשובות! 💙\n\nרוצה לשתף במשפט או שניים איך את מרגישה רגשית לאחר הלידה? זה יעזור לי להבין טוב יותר את המצב שלך."
            else:
                # Not a clear answer - use NLP to understand and guide
                if self.langchain_agent is not None:
                    # Use LangChain agent for intelligent response
                    try:
                        sentiment_tool = SentimentAnalysisTool()
                        sentiment_result = sentiment_tool._run(user_message)
                        
                        # Determine if they're sharing emotions or answering question
                        if any(kw in user_message for kw in DISTRESS_KEYWORDS):
                            response = f"אני מבינה שאת מרגישה {sentiment_result}. תודה על השיתוף 💙\n\nבואי נמשיך - {EPDS_QUESTIONS[self.state.current_question_index]}\nאנא עני עם מספר 0-3:"
                        else:
                            response = f"לא הבנתי את התשובה. אנא עני על השאלה עם מספר 0-3:\n{EPDS_QUESTIONS[self.state.current_question_index]}"
                    except:
                        response = f"אנא עני עם מספר 0-3 על השאלה:\n{EPDS_QUESTIONS[self.state.current_question_index]}"
                else:
                    response = f"אנא עני עם מספר 0-3 על השאלה:\n{EPDS_QUESTIONS[self.state.current_question_index]}"
        
        elif self.state.needs_free_text and not self.state.free_text_collected:
            # Collecting free text
            self.state.free_text = user_message
            self.state.free_text_collected = True
            
            # Analyze sentiment
            sentiment, keywords = self._analyze_sentiment(user_message)
            
            # Save assessment
            record_id, total_score = save_epds_assessment(
                self.state,
                sentiment,
                keywords
            )
            
            # Determine risk level
            risk_assessment = self._assess_risk(total_score, sentiment, keywords)
            
            # Generate response
            response = f"תודה רבה על השיתוף הכנה 💙\n\n"
            response += f"📊 תוצאות ההערכה:\n"
            response += f"ציון EPDS: {total_score}/30\n"
            response += f"רמת סיכון: {risk_assessment['risk_level']}\n"
            response += f"{risk_assessment['recommendation']}\n\n"
            
            if self.ppd_agent is not None:
                response += f"💡 רוצה שאעריך את הסיכון שלך גם עם מודל XGBoost המתקדם?\n"
                response += f"אם כן, ספרי לי קצת על הסימפטומים שלך (שינה, תיאבון, חרדה וכו')."
            
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

