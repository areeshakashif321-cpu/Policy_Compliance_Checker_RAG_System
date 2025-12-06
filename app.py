"""
Policy Compliance RAG System - Streamlit Application
"


# Page configuration
st.set_page_config(
    page_title="Policy Compliance RAG System",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources(api_key):
    """Load all necessary resources"""
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Load vector store
    vectorstore = FAISS.load_local(
        "models/vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Initialize LLM
    llm = GeminiLLM(api_key=api_key)
    
    # Initialize compliance checker
    checker = ComplianceChecker(
        vectorstore=vectorstore,
        llm=llm.model,
        rules_path="data/compliance_rules.json"
    )
    
    return vectorstore, llm, checker

@st.cache_data
def load_data():
    """Load analysis results"""
    
    # Load comparison data
    comparison_df = pd.read_csv("compliance_comparison.csv")
    
    # Load detailed results
    with open("detailed_compliance_results.json", 'r') as f:
        detailed_results = json.load(f)
    
    # Load compliance rules
    with open("compliance_rules.json", 'r') as f:
        compliance_rules = json.load(f)
    
    return comparison_df, detailed_results, compliance_rules

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">📋 Policy Compliance RAG System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/legal-document.png", width=100)
        st.title("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Google API Key",
            type="password",
            help="Enter your Google Gemini API key"
        )
        
        if not api_key:
            st.warning("⚠️ Please enter your API key to continue")
            st.stop()
        
        st.success("✅ API Key configured")
        
        # Navigation
        st.markdown("---")
        st.title("📑 Navigation")
        page = st.radio(
            "Select Page",
            ["🏠 Dashboard", "🔍 Compliance Checker", "💬 Q&A System", "📊 Analytics", "📋 Rules Explorer"]
        )
    
    # Load resources
    try:
        vectorstore, llm, checker = load_resources(api_key)
        comparison_df, detailed_results, compliance_rules = load_data()
    except Exception as e:
        st.error(f"❌ Error loading resources: {str(e)}")
        st.info("💡 Make sure all data files are in the correct folders")
        st.stop()
    
    # Page routing
    if page == "🏠 Dashboard":
        show_dashboard(comparison_df, detailed_results, compliance_rules)
    
    elif page == "🔍 Compliance Checker":
        show_compliance_checker(checker)
    
    elif page == "💬 Q&A System":
        show_qa_system(checker)
    
    elif page == "📊 Analytics":
        show_analytics(comparison_df, detailed_results)
    
    elif page == "📋 Rules Explorer":
        show_rules_explorer(compliance_rules)

def show_dashboard(comparison_df, detailed_results, compliance_rules):
    """Dashboard page"""
    
    st.header("📊 Executive Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Contracts",
            value=len(comparison_df),
            delta=None
        )
    
    with col2:
        compliant_count = len(comparison_df[comparison_df['Total Violations'] == 0])
        st.metric(
            label="Fully Compliant",
            value=compliant_count,
            delta=f"{compliant_count/len(comparison_df)*100:.1f}%"
        )
    
    with col3:
        avg_compliance = comparison_df['Compliance %'].str.rstrip('%').astype(float).mean()
        st.metric(
            label="Avg Compliance",
            value=f"{avg_compliance:.1f}%",
            delta=None
        )
    
    with col4:
        total_violations = comparison_df['Total Violations'].sum()
        st.metric(
            label="Total Violations",
            value=total_violations,
            delta=None
        )
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Compliance Distribution")
        fig = px.histogram(
            comparison_df,
            x='Compliance %',
            nbins=20,
            title="Distribution of Compliance Scores",
            labels={'Compliance %': 'Compliance Percentage'},
            color_discrete_sequence=['#1f77b4']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Compliance Status")
        status_counts = comparison_df['Status'].value_counts()
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Overall Compliance Status",
            color_discrete_sequence=['#2ecc71', '#e74c3c']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Severity breakdown
    st.subheader("⚠️ Violations by Severity")
    
    severity_data = pd.DataFrame({
        'Severity': ['High', 'Medium', 'Low'],
        'Count': [
            comparison_df['High Severity'].sum(),
            comparison_df['Medium Severity'].sum(),
            comparison_df['Low Severity'].sum()
        ]
    })
    
    fig = px.bar(
        severity_data,
        x='Severity',
        y='Count',
        title="Total Violations by Severity Level",
        color='Severity',
        color_discrete_map={'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#f1c40f'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent contracts table
    st.subheader("📋 Recent Compliance Analysis")
    display_df = comparison_df[['Filename', 'Compliance %', 'Total Violations', 'Status']].head(10)
    st.dataframe(display_df, use_container_width=True)

def show_compliance_checker(checker):
    """Compliance checker page"""
    
    st.header("🔍 Compliance Checker")
    st.write("Ask questions about contract compliance and get detailed analysis.")
    
    # Predefined queries
    st.subheader("💡 Suggested Queries")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Check Party Identification"):
            query = "Are the contracting parties clearly identified?"
            st.session_state.compliance_query = query
        
        if st.button("Check Effective Dates"):
            query = "Do contracts specify effective dates?"
            st.session_state.compliance_query = query
        
        if st.button("Check Termination Clauses"):
            query = "What are the termination provisions in the contracts?"
            st.session_state.compliance_query = query
    
    with col2:
        if st.button("Check Governing Law"):
            query = "Are governing law clauses present?"
            st.session_state.compliance_query = query
        
        if st.button("Check Liability Caps"):
            query = "Do contracts define liability caps?"
            st.session_state.compliance_query = query
        
        if st.button("Check IP Ownership"):
            query = "Are IP ownership terms clearly defined?"
            st.session_state.compliance_query = query
    
    st.markdown("---")
    
    # Custom query
    st.subheader("✍️ Custom Compliance Query")
    
    query = st.text_area(
        "Enter your compliance question:",
        value=st.session_state.get('compliance_query', ''),
        height=100,
        placeholder="e.g., Are renewal terms clearly specified in the contracts?"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        check_button = st.button("🔍 Check Compliance", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.session_state.compliance_query = ''
        st.rerun()
    
    if check_button and query:
        with st.spinner("🔍 Analyzing contracts..."):
            try:
                result = checker.check_compliance(query)
                
                st.success("✅ Analysis Complete!")
                
                # Display results
                st.subheader("📋 Compliance Analysis")
                st.markdown(result['response'])
                
                # Display sources
                with st.expander("📄 View Sources"):
                    st.write(f"**Documents Analyzed:** {result['num_sources']}")
                    st.write("**Source Files:**")
                    for i, source in enumerate(result['sources'][:5], 1):
                        st.write(f"{i}. {source}")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

def show_qa_system(checker):
    """Q&A system page"""
    
    st.header("💬 Contract Q&A System")
    st.write("Ask any question about the contracts and get instant answers.")
    
    # Chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if question := st.chat_input("Ask a question about the contracts..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = checker.answer_question(question)
                    response = result['answer']
                    
                    st.markdown(response)
                    
                    # Show sources
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(result['sources'][:3], 1):
                            st.write(f"{i}. {source}")
                    
                    # Add assistant message
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

def show_analytics(comparison_df, detailed_results):
    """Analytics page"""
    
    st.header("📊 Detailed Analytics")
    
    # Tabs for different analytics
    tab1, tab2, tab3 = st.tabs(["📈 Trends", "🔝 Top Violators", "📉 Severity Analysis"])
    
    with tab1:
        st.subheader("Compliance Trends")
        
        # Compliance distribution
        compliance_values = comparison_df['Compliance %'].str.rstrip('%').astype(float)
        
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=compliance_values,
            name='Compliance %',
            marker_color='lightblue'
        ))
        fig.update_layout(
            title="Compliance Score Distribution (Box Plot)",
            yaxis_title="Compliance %"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Median", f"{compliance_values.median():.1f}%")
        with col2:
            st.metric("Mean", f"{compliance_values.mean():.1f}%")
        with col3:
            st.metric("Std Dev", f"{compliance_values.std():.1f}%")
    
    with tab2:
        st.subheader("Top 10 Most Non-Compliant Contracts")
        
        top_violators = comparison_df.nlargest(10, 'Total Violations')
        
        fig = px.bar(
            top_violators,
            x='Total Violations',
            y='Filename',
            orientation='h',
            title="Contracts with Most Violations",
            color='Total Violations',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.dataframe(
            top_violators[['Filename', 'Compliance %', 'Total Violations', 'High Severity', 'Medium Severity', 'Low Severity']],
            use_container_width=True
        )
    
    with tab3:
        st.subheader("Violation Severity Breakdown")
        
        # Stacked bar chart
        top_10 = comparison_df.nlargest(10, 'Total Violations')
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='High',
            x=top_10['Filename'],
            y=top_10['High Severity'],
            marker_color='#e74c3c'
        ))
        fig.add_trace(go.Bar(
            name='Medium',
            x=top_10['Filename'],
            y=top_10['Medium Severity'],
            marker_color='#f39c12'
        ))
        fig.add_trace(go.Bar(
            name='Low',
            x=top_10['Filename'],
            y=top_10['Low Severity'],
            marker_color='#f1c40f'
        ))
        
        fig.update_layout(
            barmode='stack',
            title='Severity Breakdown (Top 10 Violators)',
            xaxis_tickangle=-45,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

def show_rules_explorer(compliance_rules):
    """Rules explorer page"""
    
    st.header("📋 Compliance Rules Explorer")
    
    # Filter by severity
    severity_filter = st.multiselect(
        "Filter by Severity",
        options=['HIGH', 'MEDIUM', 'LOW'],
        default=['HIGH', 'MEDIUM', 'LOW']
    )
    
    # Display rules
    for rule_id, rule in compliance_rules.items():
        if rule['severity'] in severity_filter:
            severity_color = {
                'HIGH': '🔴',
                'MEDIUM': '🟡',
                'LOW': '🟢'
            }
            
            with st.expander(f"{severity_color[rule['severity']]} {rule_id}: {rule['name']}"):
                st.write(f"**Description:** {rule['description']}")
                st.write(f"**Severity:** {rule['severity']}")
                st.write(f"**Check:** {rule['check']}")
                st.write(f"**Remediation:** {rule['remediation']}")
                
                if rule['related_columns']:
                    st.write(f"**Related Columns:** {', '.join(rule['related_columns'])}")

if __name__ == "__main__":
    main()
