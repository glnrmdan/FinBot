---
title: Chatbot Finance
emoji: 🐢
colorFrom: green
colorTo: gray
sdk: streamlit
sdk_version: 1.41.1
app_file: app.py
pinned: false
---

You can access the App on:
https://finbot-cl7jxrpbsx6klg5juwiqrp.streamlit.app/

<h1>How To Use</h1>
<ul>
  <li>1. Open the link above</li>
  <li>2. Upload the PDF Files of Financial Reports (You can use the sample reports from this repo, just download it)</li>
  <li>3. Wait until the analysis process done</li>
  <li>4. Ask any question that you want to ask. For example : I want to know the net profit of the company</li>
</ul>

___
Testing Agent:
```
# Agent classes
class FinancialAnalyst:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate.from_template(
            "As an expert financial analyst, analyze this financial report:\n\n"
            "{report}\n\n"
            "Provide a detailed analysis covering:\n"
            "1. Revenue and profit trends\n"
            "2. Balance sheet health\n"
            "3. Cash flow analysis\n"
            "4. Key financial ratios\n"
            "5. Comparison with industry benchmarks\n"
            "Use specific numbers from the report in your analysis."
        )

    def analyze(self, report):
        chain = self.prompt | self.llm | StrOutputParser()
        return "".join(chain.stream({"report": report}))

class ManagerFinancialAnalyst:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate.from_template(
            "As a senior financial manager, review this analysis:\n\n"
            "{analysis}\n\n"
            "Provide a critical review addressing:\n"
            "1. Accuracy of the analysis\n"
            "2. Completeness of the assessment\n"
            "3. Potential risks or opportunities overlooked\n"
            "4. Additional insights based on market knowledge\n"
            "5. Recommendations for further analysis if needed"
        )

    def review(self, analysis):
        chain = self.prompt | self.llm | StrOutputParser()
        return chain.stream({"analysis": analysis})

class FinanceConsultant:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate.from_template(
            "As an experienced finance consultant, consider this analysis and review:\n\n"
            "Analysis:\n{analysis}\n\n"
            "Review:\n{review}\n\n"
            "Provide comprehensive investment recommendations:\n"
            "1. Overall investment stance (bullish, bearish, or neutral)\n"
            "2. Specific investment strategies\n"
            "3. Potential risks and mitigation strategies\n"
            "4. Short-term and long-term investment outlook\n"
            "5. Diversification suggestions\n"
            "Justify your recommendations based on the provided information."
        )

    def recommend(self, analysis, review):
        chain = self.prompt | self.llm | StrOutputParser()
        return chain.stream({"analysis": analysis, "review": review})

# Comprehensive analysis function
@safe_api_call
def comprehensive_analysis(financial_report, llm):
    analyst = FinancialAnalyst(llm)
    manager = ManagerFinancialAnalyst(llm)
    consultant = FinanceConsultant(llm)

    analysis_placeholder = st.empty()
    review_placeholder = st.empty()
    recommendation_placeholder = st.empty()

    with st.spinner("Generating Financial Analysis..."):
        analysis = ""
        for chunk in analyst.analyze(financial_report):
            analysis += chunk
            analysis_placeholder.markdown(analysis)

    with st.spinner("Generating Manager's Review..."):
        review = ""
        for chunk in manager.review(analysis):
            review += chunk
            review_placeholder.markdown(review)

    with st.spinner("Generating Investment Recommendation..."):
        recommendation = ""
        for chunk in consultant.recommend(analysis, review):
            recommendation += chunk
            recommendation_placeholder.markdown(recommendation)

    return {"analysis": analysis, "review": review, "recommendation": recommendation}
```
