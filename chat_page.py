import streamlit as st
import tempfile

def show():
    st.title("Document Q&A Chat Interface")
    
    st.write("Chat with your documents using one or more language models.")
    
    # Document upload
    uploaded_file = st.file_uploader("Choose a document file", type=["txt", "pdf", "doc", "docx"])
    
    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        try:
            embeddings = st.session_state.embeddings
            
            with st.spinner("Processing document..."):
                texts = st.session_state.process_document(temp_file_path, file_type, _embeddings=embeddings)
                db = st.session_state.create_vector_store(texts, embeddings)
                retriever = db.as_retriever(search_kwargs={"k": 3})
            
            st.success("Document processed successfully!")
            
            # Initialize chat history and models
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            if "model_chains" not in st.session_state:
                st.session_state.model_chains = {
                    model_name: st.session_state.create_conversational_chain(st.session_state.load_model(model_name), retriever)
                    for model_name in st.session_state.selected_llms
                }
            
            # Display chat messages from history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "model_responses" in message:
                        for model, response in message["model_responses"].items():
                            st.markdown(f"**{model}**: {response['answer']}")
                            st.markdown(f"Response time: {response['time']:.2f} seconds")
            
            # React to user input
            if prompt := st.chat_input("What would you like to know about the document?"):
                st.session_state.interaction_count += 1
                
                # Display user message
                st.chat_message("user").markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Get responses from all selected models
                model_responses = {}
                for model_name in st.session_state.selected_llms:
                    with st.spinner(f"Getting response from {model_name}..."):
                        answer, response_time = st.session_state.chat_with_doc(st.session_state.model_chains[model_name], prompt)
                        model_responses[model_name] = {"answer": answer, "time": response_time}
                
                # Display assistant responses
                with st.chat_message("assistant"):
                    for model_name, response in model_responses.items():
                        st.markdown(f"**{model_name}**:")
                        st.markdown(response["answer"])
                        st.markdown(f"Response time: {response['time']:.2f} seconds")
                
                # Add assistant responses to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Model Responses",
                    "model_responses": model_responses
                })
                
                # Calculate metrics and update analytics
                for model_name, response in model_responses.items():
                    metrics = st.session_state.calculate_metrics(response["answer"], retriever.get_relevant_documents(prompt), response["time"], embeddings)
                    metrics['interaction'] = st.session_state.interaction_count
                    if model_name not in st.session_state.analytics:
                        st.session_state.analytics[model_name] = []
                    st.session_state.analytics[model_name].append(metrics)
                
                st.rerun()
        
        except Exception as e:
            st.error(f"An error occurred while processing the document: {str(e)}")
            st.info("If you're experiencing issues, please make sure all required libraries are installed and up to date.")
        
        finally:
            import os
            os.unlink(temp_file_path)
    else:
        st.info("Please upload a document to start chatting.")