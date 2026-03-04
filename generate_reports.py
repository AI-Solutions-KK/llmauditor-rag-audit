#!/usr/bin/env python3
"""
Simple script to generate real LLMAuditor reports for README demonstration
"""

import os
from dotenv import load_dotenv
from llmauditor import auditor, BudgetExceededError, LowConfidenceError

# Load environment
load_dotenv()

def generate_sample_reports():
    """Generate real llmauditor reports to capture for README"""
    
    print("🔍 Generating Real LLMAuditor Reports for README...")
    print("=" * 60)
    
    # Setup auditor
    auditor.set_budget(0.50)
    auditor.guard_mode(confidence_threshold=70)
    auditor.set_alert_mode(True)
    auditor.start_evaluation("HR Knowledge RAG Demo", version="1.0.0")
    
    # Report 1: High-quality response
    print("\n📊 EXAMPLE 1: High-Quality RAG Response")
    print("-" * 40)
    report1 = auditor.execute(
        model="gpt-4o-mini",
        input_tokens=95,
        output_tokens=48,
        raw_response="Employees with 3-5 years experience receive 20 vacation days per year according to our leave policy.",
        input_text="How many vacation days do employees get after 3 years?"
    )
    report1.display()
    
    # Report 2: Intentionally poor response to trigger hallucination detection  
    print("\n\n🚨 EXAMPLE 2: Detected Hallucination (Poor Response)")
    print("-" * 50)
    try:
        report2 = auditor.execute(
            model="gpt-4o-mini",
            input_tokens=42,
            output_tokens=22,
            raw_response="Yes, cryptocurrency trading is encouraged during lunch breaks and we provide free crypto wallets.",
            input_text="What is our cryptocurrency policy?"
        )
        report2.display()
    except LowConfidenceError as e:
        print(f"🛡️ BLOCKED by guard mode: {e}")
    
    # Report 3: Budget tracking display
    print("\n\n💰 EXAMPLE 3: Budget Tracking")
    print("-" * 30)
    status = auditor.get_budget_status()
    print(f"💰 Budget Status: ${status['cumulative_cost']:.6f} spent / ${status['budget_limit']:.2f} limit ({status['cumulative_cost']/status['budget_limit']*100:.1f}% used)")
    print(f"📊 Queries processed: {status['executions']}")
    print(f"📈 Average cost: ${status['cumulative_cost']/max(1, status['executions']):.6f} per query")
    
    # Generate final certification report
    print("\n\n🏆 EXAMPLE 4: Certification Report")
    print("-" * 35)
    auditor.end_evaluation()
    eval_report = auditor.generate_evaluation_report()
    eval_report.display()
    
    # Export reports 
    try:
        os.makedirs("reports", exist_ok=True)
        paths = eval_report.export_all(output_dir="reports")
        print(f"\n📋 Reports exported to:")
        print(f"   PDF: {paths['pdf']}")
        print(f"   HTML: {paths['html']}")
        print(f"   MD: {paths['md']}")
        
        # Also export one execution report
        report1.export("html", output_dir="reports")
        print(f"   Execution: reports/audit_{report1.execution_id}.html")
        
    except Exception as e:
        print(f"Export error: {e}")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in .env file")
        exit(1)
    
    generate_sample_reports()