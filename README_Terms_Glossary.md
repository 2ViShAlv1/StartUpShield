# StartupShield AI — Terms Glossary (Hinglish)

*Ye file ek dictionary jaisi hai — project me jitne bhi technical terms use honge, sabka simple Hinglish explanation yaha milega. Jab bhi koi term bhool jaye ya confuse ho, yahi file khol lena.*

---

## Business / Domain Terms

**Churn**
Jab koi customer product/service use karna band kar deta hai (subscription cancel, account close). "Churn prediction" matlab ye predict karna ki kaunsa customer aage jaake chhod sakta hai.

**Churn Rate**
Kitne % customers ek time period (jaise ek mahina) me chhod gaye. Formula: `(chhode hue customers / total customers) × 100`.

**MRR (Monthly Recurring Revenue)**
Har mahine ka fixed/predictable revenue jo subscriptions se aata hai. SaaS companies ka health-check number.

**DAU / MAU (Daily/Monthly Active Users)**
Kitne unique users ne product ko ek din/mahine me use kiya. Company ka "engagement health" batata hai.

**Support Ticket**
Jab customer koi problem/complaint/query submit karta hai support team ko — email, chat, ya ticketing system ke through.

**Retention**
Customer ko rokna (opposite of churn). "Retention campaign" matlab customer ko rokne ke liye special offer/outreach.

**RFM (Recency, Frequency, Monetary)**
Customer behavior measure karne ka tarika: Recency (kitne din pehle last use kiya), Frequency (kitni baar use karta hai), Monetary (kitna paisa spend karta hai). Churn prediction me common feature set hai.

**Risk Score**
Ek single number (jaise 0-100) jo batata hai ki company/customer kitna "at risk" hai — jitna high, utna zyada danger.

**Runway**
Startup ke paas kitne mahine ka paisa bacha hai current spending rate pe (bank balance ÷ monthly burn).

---

## Machine Learning — General Terms

**Supervised Learning**
Jab model ko labeled data diya jata hai (input + correct answer dono), aur wo seekhta hai input se answer predict karna. Jaise: customer data + "churned: yes/no" label.

**Unsupervised Learning**
Jab model ko sirf input data diya jata hai, koi label nahi — model khud patterns dhundta hai. Jaise: Isolation Forest bina bataye anomalies dhund leta hai.

**Classification**
Predict karna ki data kaunse "category/class" me aata hai. Jaise: churn = yes/no, sentiment = positive/negative/neutral.

**Regression**
Ek continuous number predict karna (category nahi). Jaise: agle mahine ka revenue kitna hoga.

**Feature**
Model ko diya gaya har input column/signal. Jaise churn model me tenure, monthly_spend — ye sab "features" hain.

**Label / Target**
Wo cheez jo model ko predict karni hai. Jaise churn_label (0 ya 1).

**Training**
Model ko data dikha ke usse patterns seekhne dena — weights adjust hote hain taaki predictions sahi ho.

**Overfitting**
Jab model training data ko itna "rat-lo" (memorize) kar leta hai ki naye/unseen data pe achha perform nahi karta. Jaise exam me sirf notes ratt lena, samajhna nahi.

**Underfitting**
Jab model itna simple hai ki wo data ke patterns hi nahi pakad pa raha — training aur test dono pe bekar perform karta hai.

**Cross-Validation**
Data ko multiple chunks (folds) me baant ke model ko baar-baar train/test karna, taaki evaluation reliable ho — ek hi split pe depend na kare.

**Hyperparameter Tuning**
Model ke "settings" (jaise tree depth, learning rate) ko adjust karke best combination dhundna.

**Data Leakage**
Jab training data me galti se aisi info aa jati hai jo real-world me predict karte time available nahi hoti — model artificially achha perform karta hai but production me fail hota hai. Jaise "cancellation_date" ko churn predict karne ke liye use karna.

**Imbalanced Data**
Jab ek class dusri se bahut kam hoti hai (jaise 90% "no churn", 10% "churn"). Model bias ho sakta hai majority class ki taraf.

**Class Weight / SMOTE**
Imbalanced data handle karne ke tarike — class_weight model ko minority class pe zyada dhyan dene ko kehta hai, SMOTE synthetic minority samples generate karta hai.

---

## Model Names (Specific)

**Logistic Regression**
Simplest classification model — features ko weight de ke ek probability nikalta hai. Fast, interpretable, achha baseline.

**Random Forest**
Bahut saare decision trees banata hai aur unka average/vote leta hai. Robust aur accurate, tabular data pe achha kaam karta hai.

**XGBoost**
"Boosting" technique — har naya tree pichle tree ki galti fix karta hai. Tabular data competitions me sabse popular model.

**Isolation Forest**
Anomaly detection ka model — jo data point jaldi "isolate" ho jata hai (kam splits me alag), usse anomaly maana jata hai.

**Contamination**
Isolation Forest ka parameter — expected % anomalies dataset me (jaise 0.03 = 3%). Ye seedha precision ko control karta hai: jitna kam contamination, utne kam flags.

**Autoencoder**
Neural network jo data ko compress karke phir wapas reconstruct karta hai — agar reconstruction galat/off hai, toh wo data anomaly ho sakta hai.

**TF-IDF (Term Frequency-Inverse Document Frequency)**
Text ko numbers me convert karne ka tarika — rare aur important words ko zyada weight deta hai (common words jaise "the" ko kam).

**BERT / DistilBERT**
Transformer-based pretrained language model jo English "samajhta" hai (context ke saath). DistilBERT BERT ka chhota, fast version hai. Pretrained hota hai, fir apne data pe fine-tune karte hain.

**LSTM (Long Short-Term Memory)**
Ek type ka neural network jo sequences/time-series ka pattern yaad rakh ke agla value predict karta hai — jaise revenue history dekh ke agla mahina predict karna.

**Prophet**
Facebook ka banaya forecasting tool — trend aur seasonality automatically pakad leta hai, kam tuning me achha result deta hai.

**LightGBM**
XGBoost jaisa hi gradient boosting model, but faster aur kam memory leta hai bade data pe. StartupShield me churn ka best model yahi hai.

**HistGradientBoosting**
Scikit-learn ka apna gradient boosting model (LightGBM na ho toh fallback ke roop me use hota hai).

**ETS / Holt-Winters (Exponential Smoothing)**
Forecasting ka classical statistical model — trend aur seasonality ko "smoothing" se pakadta hai. Prophet na ho toh fallback.

---

## NLP (Text) Terms

**Tokenization**
Text ko chhote pieces (words/subwords) me todna taaki model process kar sake.

**Stopwords**
Common words jo zyada meaning nahi dete (jaise "is", "the", "and") — inko aksar remove kiya jata hai text cleaning me.

**Stemming / Lemmatization**
Words ko unke root form me convert karna. Jaise "running" → "run".

**Word Embedding**
Words ko numbers (vectors) me represent karne ka tarika jisme similar-meaning words ke vectors bhi close hote hain.

**Sentiment Analysis**
Text (review/ticket) ka tone/emotion detect karna — positive, negative, ya neutral.

---

## Time-Series Terms

**Trend**
Data ka overall long-term direction — upar ja raha hai, neeche, ya stable.

**Seasonality**
Repeating pattern ek fixed interval pe — jaise weekends pe kam sales, December me spike.

**Stationarity**
Jab time series ke statistical properties (mean, variance) time ke saath change nahi hote — kai forecasting models ko is assumption ki zarurat hoti hai.

**Lag Feature**
Purane values ko feature ki tarah use karna — jaise "kal ka revenue" aaj ke prediction ke liye feature ban sakta hai.

**Rolling Average**
Ek moving window (jaise last 7 din) ka average — noise kam karke trend clearly dikhata hai.

**Chronological Split**
Time-series data ko train/test me todne ka sahi tarika — hamesha time ke order me (last N din test ke liye), kabhi random shuffle nahi. Shuffle karne se "future" data "past" ko predict karne me leak ho jata hai.

**Naive Baseline**
Sabse simple forecast: "kal jo hua wahi aaj bhi hoga" (last value carry forward). Har real model ko isse behtar hona chahiye, warna wo useless hai.

**Confidence Interval / Prediction Interval**
Ek range jisme actual value ke aane ki ummeed hai (jaise "95% confident ki revenue 20k-28k ke beech hoga"). Sirf ek number dene se behtar hai kyunki uncertainty bhi dikhata hai.

**Coverage**
Prediction interval kitni baar sach me actual value ko capture karta hai. 95% interval claim kiya hai toh ~95% actuals andar aane chahiye — kam hua toh interval "over-confident" hai.

**Additive vs Multiplicative Seasonality**
Additive: seasonal swing hamesha same fixed amount hota hai. Multiplicative: swing revenue/level ke saath badhta hai (bada business, bada swing).

---

## Explainable AI (XAI) Terms

**SHAP (SHapley Additive exPlanations)**
Ek technique jo batati hai ki model ki prediction me har feature ne kitna contribute kiya — "kyu" is prediction pe pahuche, ye samjhata hai.

**Feature Importance**
Overall, kaunse features model ke liye sabse zyada matter karte hain (poore dataset ke across).

**Local Explanation**
Ek single prediction ke liye explanation ("is specific customer ke liye ye factors important the").

**Global Explanation**
Poore model ke behavior ka overall explanation (sab predictions ke across pattern).

**Interpretability**
Kitna easily insaan model ke decision ko samajh sakta hai — Logistic Regression highly interpretable hai, deep neural nets kam.

---

## Evaluation Metrics

**Accuracy**
Kitne % predictions sahi the (overall) — imbalanced data me misleading ho sakta hai.

**Precision**
Jab model ne "positive" bola, kitni baar sahi tha. (False alarms kam karne pe focus.)

**Recall**
Actual positives me se kitne model ne pakde. (Miss kam karne pe focus.)

**F1 Score**
Precision aur Recall ka balance (harmonic mean) — jab dono important hon.

**ROC-AUC**
Model kitna achha positive aur negative classes ko differentiate kar pa raha hai (0.5 = random guessing, 1.0 = perfect).

**PR-AUC (Precision-Recall AUC)**
ROC-AUC jaisa hi but imbalanced data ke liye zyada reliable metric.

**MAE (Mean Absolute Error)**
Forecast aur actual value ke beech average difference (simple, easy to interpret).

**RMSE (Root Mean Squared Error)**
MAE jaisa hi but bade errors ko zyada penalize karta hai.

**MAPE (Mean Absolute Percentage Error)**
Error ko percentage me batata hai — samajhne me aasan ("average 8% off tha").

---

## MLOps / Deployment Terms

**API (Application Programming Interface)**
Ek "pul" jisse do software ek dusre se baat karte hain. Jaise dashboard, model se prediction "API call" karke maangta hai.

**FastAPI**
Python framework jisse fast, modern REST APIs banate hain.

**Streamlit**
Python library jisse bina web-development knowledge ke interactive dashboards/apps bana sakte hain.

**Model Serialization**
Trained model ko file me save karna (jaise `.pkl`) taaki baad me/kahin aur use ho sake, dobara train na karna pade.

**Docker**
Tool jo application ko uske dependencies ke saath ek "container" me pack karta hai — kisi bhi machine pe same tarike se chalta hai.

**MLflow**
Tool jo ML experiments (kaunsa model, kaunse parameters, kya result) track karta hai — comparison aasan hota hai.

**CI/CD (Continuous Integration/Continuous Deployment)**
Automated process jo code changes ko test aur deploy karta hai bina manually har baar karna pade.

**Data Validation**
Ye check karna ki incoming data expected format/range me hai, corrupt/garbage nahi hai — production me bugs se bachata hai.

---

## Project-Specific Terms (StartupShield AI)

**Risk Aggregator**
Wo module jo churn, sentiment, anomaly, aur forecast — sab outputs ko combine karke ek final risk score banata hai.

**Recommendation Engine**
Wo module jo risk drivers dekh ke actionable suggestions deta hai (jaise "retention outreach karo").

**Baseline Model**
Sabse simple model jo pehle banate hain comparison ke liye — taaki pata chale complex model actually value add kar raha hai ya nahi.

**MVP (Minimum Viable Product)**
Project ka sabse chhota useful version jo core value dikhata hai bina saari advanced features ke.

---

*Bas — jab bhi kisi term pe atke, ye file check kar lena. Agar koi naya term aaye jo yaha nahi hai, bata dena, add kar dunga.*
