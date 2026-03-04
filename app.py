#!/usr/bin/env python3
"""
RAG Pipeline Auditor - Real-world HR Knowledge Base with LLMAuditor Integration

This application demonstrates proper LLMAuditor usage in a production RAG system.
It intentionally includes some imperfect scenarios to showcase auditing capabilities.

Usage:
    python app.py

Author: AI-Solutions-KK
License: MIT
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# LLMAuditor - the star of the show!
from llmauditor import auditor, BudgetExceededError, LowConfidenceError

# Initialize console
console = Console()

# Load environment variables
load_dotenv()

class HRKnowledgeRAG:
    """RAG system for HR knowledge base with comprehensive LLMAuditor integration."""
    
    def __init__(self):
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.chain = None
        self.setup_auditor()
        
    def setup_auditor(self):
        """Configure LLMAuditor for production RAG monitoring."""
        # Set budget limit for the demo session
        budget_limit = float(os.getenv("LLMAUDITOR_BUDGET_LIMIT", "0.50"))
        auditor.set_budget(budget_limit)
        
        # Configure guard mode to catch low-confidence responses  
        confidence_threshold = int(os.getenv("LLMAUDITOR_CONFIDENCE_THRESHOLD", "70"))
        auditor.guard_mode(confidence_threshold=confidence_threshold)
        
        # Enable alert mode instead of crashing on budget issues
        auditor.set_alert_mode(True)
        
        # Start evaluation session for this RAG system
        auditor.start_evaluation("HR Knowledge RAG", version="1.0.0")
        
        console.print(Panel(
            f"[bold green]LLMAuditor Configuration[/bold green]\n\n"
            f"💰 Budget Limit: ${budget_limit:.2f}\n"
            f"🛡️ Guard Mode: {confidence_threshold}% confidence threshold\n" 
            f"📊 Alert Mode: Enabled (warnings instead of crashes)\n"
            f"📋 Evaluation Session: Started",
            title="🔍 RAG Audit System", 
            border_style="green"
        ))
        
    def load_documents(self) -> List[Document]:
        """Load HR policy documents from the data directory."""
        documents = []
        data_path = Path("data")
        
        if not data_path.exists():
            console.print("[bold red]Error: data/ directory not found![/bold red]")
            sys.exit(1)
            
        for txt_file in data_path.glob("*.txt"):
            content = txt_file.read_text(encoding='utf-8')
            documents.append(Document(
                page_content=content,
                metadata={"source": txt_file.name, "type": "hr_policy"}
            ))
            
        console.print(f"[bold blue]📄 Loaded {len(documents)} HR policy documents[/bold blue]")
        return documents
        
    def create_vectorstore(self, documents: List[Document]):
        """Create FAISS vector store with OpenAI embeddings."""
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        splits = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"  # Cost-effective option
        )
        
        # Create FAISS vector store
        self.vectorstore = FAISS.from_documents(splits, embeddings)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Return top 3 relevant chunks
        )
        
        console.print(f"[bold green]🔍 Created vector store with {len(splits)} document chunks[/bold green]")
        
    def setup_rag_chain(self):
        """Setup the RAG chain with intentional quality variations."""
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # Cost-effective for demo
            temperature=0.2  # Some creativity but mostly factual
        )
        
        # RAG prompt template
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful HR assistant for TechCorp Inc. 
            Answer questions based on the provided company policy context.
            
            Context:
            {context}
            
            Question: {question}
            
            Instructions:
            - Provide accurate information based on the context
            - If information is not in the context, say "I don't have information about that in our policies"
            - Be helpful but stick to company policies
            - Format your answer clearly
            
            Answer:"""
        )
        
        # Create the RAG chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
            
        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
    def create_low_quality_response(self, question: str) -> Dict[str, Any]:
        """Intentionally create a poor response to test auditing (for demo purposes)."""
        # Simulate a problematic response that should trigger hallucination detection
        poor_responses = [
            "Yes, absolutely! You can work from home whenever you want, no restrictions.",
            "The company provides unlimited vacation days and a $10,000 professional development budget.",
            "We have a bring-your-pet-to-work policy and cryptocurrency trading is encouraged during lunch.",
            "TechCorp offers free housing and unlimited personal use of company equipment.",
            "All employees receive a $5,000 signing bonus and guaranteed promotions every 6 months."
        ]
        
        return {
            "answer": poor_responses[len(question) % len(poor_responses)],
            "input_tokens": 50,  # Simulated low token count
            "output_tokens": 25,  # Short, low-quality response
            "retrieval_quality": "poor"
        }
        
    def query_rag(self, question: str, intentionally_poor: bool = False) -> Dict[str, Any]:
        """Execute RAG query with LLMAuditor tracking."""
        
        if intentionally_poor:
            # For demonstration: create obviously wrong response
            response_data = self.create_low_quality_response(question)
            answer = response_data["answer"]
            input_tokens = response_data["input_tokens"]  
            output_tokens = response_data["output_tokens"]
        else:
            # Normal RAG pipeline
            try:
                answer = self.chain.invoke(question)
                input_tokens = len(question.split()) * 1.3  # Rough estimate
                output_tokens = len(answer.split()) * 1.3   # Rough estimate
            except Exception as e:
                answer = f"Error processing query: {str(e)}"
                input_tokens = len(question.split()) * 1.3
                output_tokens = len(answer.split()) * 1.3
        
        # Audit the execution with LLMAuditor
        try:
            report = auditor.execute(
                model="gpt-4o-mini",
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens), 
                raw_response=answer,
                input_text=question
            )
            
            # Display the audit panel
            report.display()
            
            return {
                "answer": answer,
                "audit_report": report,
                "success": True
            }
            
        except BudgetExceededError as e:
            console.print(f"[bold red]💰 Budget Exceeded: {e}[/bold red]")
            return {"answer": "Budget exceeded - query blocked", "success": False}
            
        except LowConfidenceError as e:
            console.print(f"[bold yellow]🛡️ Low Confidence Response Blocked: {e}[/bold yellow]")  
            return {"answer": "Low confidence response blocked by guard mode", "success": False}
            
    def run_sample_queries(self):
        """Run a series of sample queries to demonstrate different audit scenarios."""
        
        # Load sample queries
        queries_file = Path("sample_queries.txt")
        if not queries_file.exists():
            console.print("[bold red]sample_queries.txt not found![/bold red]")
            return
            
        with open(queries_file) as f:
            content = f.read()
            
        # Extract just the questions (lines that start with numbers)
        questions = []
        for line in content.split('\n'):
            line = line.strip()
            if line and any(line.startswith(str(i) + '.') for i in range(1, 30)):
                # Remove number prefix
                question = line.split('.', 1)[-1].strip()
                if question and not question.startswith('['):  # Skip empty or bracketed items
                    questions.append(question)
        
        console.print(Panel(
            f"[bold cyan]🧪 Running {len(questions)} Sample Queries[/bold cyan]\n\n"
            "This will demonstrate various audit scenarios:\n"
            "✅ High-confidence accurate responses\n"
            "⚠️ Low-confidence or ambiguous responses\n" 
            "🚨 Potential hallucinations\n"
            "💰 Budget tracking across queries",
            title="Sample Query Execution",
            border_style="cyan"
        ))
        
        for i, question in enumerate(questions[:8]):  # Run first 8 to stay within budget
            console.print(f"\n[bold magenta]Query {i+1}:[/bold magenta] {question}")
            
            # Occasionally inject poor responses for demo
            is_poor = (i == 3 or i == 6)  # Make queries 4 and 7 intentionally poor
            result = self.query_rag(question, intentionally_poor=is_poor)
            
            if result["success"]:
                console.print(f"[green]✅ Answer:[/green] {result['answer'][:100]}...")
            else:
                console.print(f"[red]❌ {result['answer']}[/red]")
                
            # Show budget status
            status = auditor.get_budget_status()
            console.print(
                f"[dim]💰 Budget: ${status['cumulative_cost']:.4f} / "
                f"${status['budget_limit']:.2f} ({status['executions']} queries)[/dim]"
            )
            
            if status['cumulative_cost'] >= status['budget_limit'] * 0.9:
                console.print("[bold yellow]⚠️ Approaching budget limit![/bold yellow]")
                
    def interactive_mode(self):
        """Interactive query mode for testing."""
        console.print(Panel(
            "[bold green]🤖 Interactive HR Knowledge Base[/bold green]\n\n"
            "Ask questions about TechCorp policies!\n"
            "- Type 'quit' to exit\n"
            "- Type 'budget' to check budget status\n"
            "- Type 'report' to generate certification report\n"
            "- Type 'poor:<question>' to simulate poor response",
            title="Interactive Mode",
            border_style="green"
        ))
        
        while True:
            question = Prompt.ask("\n[bold cyan]Your question")
            
            if question.lower() in ['quit', 'exit']:
                break
            elif question.lower() == 'budget':
                status = auditor.get_budget_status()
                console.print(Panel(
                    f"Budget Limit: ${status['budget_limit']:.2f}\n"
                    f"Spent: ${status['cumulative_cost']:.4f}\n" 
                    f"Remaining: ${status['remaining']:.4f}\n"
                    f"Executions: {status['executions']}",
                    title="💰 Budget Status"
                ))
                continue
            elif question.lower() == 'report':
                self.generate_certification_report()
                continue
                
            # Check for intentionally poor response request
            is_poor = question.startswith('poor:')
            if is_poor:
                question = question[5:].strip()
                
            result = self.query_rag(question, intentionally_poor=is_poor)
            
            if result["success"]:
                console.print(Panel(result["answer"], title="🤖 HR Assistant Response", border_style="blue"))
            else:
                console.print(f"[red]{result['answer']}[/red]")
                
    def generate_certification_report(self):
        """Generate and export certification report."""
        try:
            # End the evaluation session
            auditor.end_evaluation()
            
            # Generate certification report  
            eval_report = auditor.generate_evaluation_report()
            eval_report.display()
            
            # Export reports
            os.makedirs("reports", exist_ok=True)
            paths = eval_report.export_all(output_dir="reports")
            
            console.print(Panel(
                f"[bold green]📊 Certification Report Generated[/bold green]\n\n"
                f"PDF: {paths['pdf']}\n"
                f"HTML: {paths['html']}\n" 
                f"Markdown: {paths['md']}\n\n"
                f"Certification Level: [bold]{eval_report.score.level}[/bold] "
                f"({eval_report.score.overall:.1f}/100)",
                title="📋 Report Export Complete",
                border_style="green"
            ))
            
        except Exception as e:
            console.print(f"[red]Error generating report: {e}[/red]")

def main():
    """Main application entry point."""
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print(Panel(
            "[bold red]❌ OpenAI API Key Required[/bold red]\n\n"
            "Please set your OpenAI API key:\n\n"
            "1. Copy .env.example to .env\n"
            "2. Add your OpenAI API key to .env\n" 
            "3. Restart the application",
            title="Configuration Error",
            border_style="red"
        ))
        sys.exit(1)
    
    console.print(Panel(
        "[bold blue]🔍 RAG Pipeline Auditor[/bold blue]\n\n"
        "HR Knowledge Base with LLMAuditor Integration\n"
        "Demonstrates real-world GenAI governance and auditing",
        title="Welcome",
        border_style="blue"
    ))
    
    # Initialize RAG system
    rag = HRKnowledgeRAG()
    
    # Load and process documents
    console.print("\n[bold yellow]📚 Setting up knowledge base...[/bold yellow]")
    documents = rag.load_documents()
    rag.create_vectorstore(documents)
    rag.setup_rag_chain()
    
    # Ask user what they want to do
    mode = Prompt.ask(
        "\nChoose mode",
        choices=["sample", "interactive", "both"],
        default="both"
    )
    
    if mode in ["sample", "both"]:
        rag.run_sample_queries()
        
    if mode in ["interactive", "both"]:
        rag.interactive_mode()
        
    # Generate final certification report
    rag.generate_certification_report()

if __name__ == "__main__":
    main()