"""
Compliance Checker Utility Functions
"""

import json
from typing import Dict, List, Any
import google.generativeai as genai

class ComplianceChecker:
    """Handles compliance checking operations"""
    
    def __init__(self, vectorstore, llm, rules_path):
        self.vectorstore = vectorstore
        self.llm = llm
        
        # Load compliance rules
        with open(rules_path, 'r') as f:
            self.compliance_rules = json.load(f)
    
    def check_compliance(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Check compliance based on query
        """
        # Retrieve relevant documents
        relevant_docs = self.vectorstore.similarity_search(query, k=top_k)
        
        # Prepare context
        context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create prompt
        prompt = f"""
You are a legal compliance expert analyzing contracts.

COMPLIANCE RULES:
{json.dumps(self.compliance_rules, indent=2)}

CONTRACT SECTIONS:
{context}

QUERY: {query}

Provide a detailed compliance analysis with:
1. COMPLIANCE STATUS: [COMPLIANT/NON-COMPLIANT/PARTIAL]
2. APPLICABLE RULES: Which rules apply
3. EVIDENCE: Specific quotes from contracts
4. VIOLATIONS: Any issues found
5. REMEDIATION: Steps to fix issues

Response:
"""
        
        # Get response
        response = self.llm.generate_content(prompt)
        
        return {
            "query": query,
            "response": response.text,
            "sources": [doc.metadata.get("filename", "Unknown") for doc in relevant_docs],
            "num_sources": len(relevant_docs)
        }
    
    def answer_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Answer general questions about contracts
        """
        # Retrieve relevant documents
        relevant_docs = self.vectorstore.similarity_search(question, k=top_k)
        
        # Prepare context
        context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create prompt
        prompt = f"""
Based on the following contract information, answer the question accurately.

CONTRACT INFORMATION:
{context}

QUESTION: {question}

Provide a clear, detailed answer with specific references.

Answer:
"""
        
        response = self.llm.generate_content(prompt)
        
        return {
            "question": question,
            "answer": response.text,
            "sources": [doc.metadata.get("filename", "Unknown") for doc in relevant_docs]
        }

class GeminiLLM:
    """Wrapper for Gemini LLM"""
    
    def __init__(self, api_key: str, model_name: str = "models/gemini-2.0-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def generate_content(self, prompt: str):
        """Generate content from prompt"""
        return self.model.generate_content(prompt)
