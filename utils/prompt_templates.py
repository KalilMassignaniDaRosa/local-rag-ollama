from typing import List, Dict

class PromptTemplates:    
    @staticmethod
    def _format_chunk(result: Dict) -> str:
        """Formats a retrieved chunk for prompt insertion"""
        lines = [
            f"[{result['chunk_id']}] (score: {result['similarity_score']:.4f})",
            f"Document: {result['document_name']}",
            f"Source ID: {result['source_id']}",
        ]
        
        if result.get('page'):
            lines.append(f"Page: {result['page']}")
        
        lines.extend([
            f"Content:\n{result['text_content']}",
            "---"
        ])
        
        return "\n".join(lines)
    
    @staticmethod
    def advanced_query_template(
        question: str, 
        retrieved_chunks: List[Dict]
    ) -> str:
        """Template for advanced RAG query"""
        # Formata todos os chunks
        injected_documents = "```\n" + "\n".join([
            PromptTemplates._format_chunk(result)
            for result in retrieved_chunks
        ]) + "\n```\n"
        
        return f"""# ADVANCED RAG SYSTEM WITH VECTOR RETRIEVAL

            ## CONTEXT
            You are a specialized assistant for analyzing documents with PRECISE CITATIONS.
            Use EXCLUSIVELY the information from the retrieved chunks below.

            ## RETRIEVED CHUNKS (ordered by relevance)
            {injected_documents}

            ## CRITICAL CITATION RULES
            1. **ALWAYS cite the chunk_id** in the format [chunk_id] for each piece of information used
            2. **DO NOT invent information** that is not in the chunks
            3. **Combine multiple chunks** when the answer requires complementary information
            4. **Indicate if the material is insufficient** to answer completely
            5. **Prioritize chunks with high scores** (>0.7) in your response

            ## RESPONSE FORMAT
            Structure your response in two parts:

            **ANSWER:**
            [Answer text with inline citations using [chunk_id]]

            **SOURCES:**
            - [chunk_id]: document_name (score X.XX)
            - [chunk_id]: document_name (score X.XX)

            ## USER QUESTION
            {question}

            Respond now:"""
    
    @staticmethod
    def document_summary_template(retrieved_chunks: List[Dict]) -> str:
        """Template for generating document summaries"""
        injected_documents = "```\n" + "\n".join([
            PromptTemplates._format_chunk(result)
            for result in retrieved_chunks
        ]) + "\n```\n"
        
        return f"""# DOCUMENT SUMMARY

            ## DOCUMENT CHUNKS
            {injected_documents}

            ## INSTRUCTIONS
            Create a structured and comprehensive summary including:

            1. **Title and Main Theme** [cite: chunk_id]
            2. **Key Points** (3-5 most important items) [cite: chunk_id]
            3. **Conclusions and Insights** [cite: chunk_id]
            4. **Additional Relevant Information** [cite: chunk_id]

            Use the citation format [chunk_id] for each piece of information.

            Generate the summary now:"""
    
    @staticmethod
    def comparison_template(
        question: str,
        retrieved_chunks: List[Dict]
    ) -> str:
        """Template for comparative analysis"""
        injected_documents = "```\n" + "\n".join([
            PromptTemplates._format_chunk(result)
            for result in retrieved_chunks
        ]) + "\n```\n"
        
        return f"""# COMPARATIVE DOCUMENT ANALYSIS

            ## RETRIEVED CHUNKS
            {injected_documents}

            ## TASK
            Compare the information in the retrieved chunks regarding: {question}

            ## INSTRUCTIONS
            1. Identify **similarities** between documents [cite: chunk_id]
            2. Identify **differences** between documents [cite: chunk_id]
            3. Highlight **unique points** from each document [cite: chunk_id]
            4. Provide a **comparative synthesis** [cite: chunk_id]

            Respond now:"""
    
    @staticmethod
    def information_extraction_template(
        query: str,
        retrieved_chunks: List[Dict]
    ) -> str:
        """Template for specific information extraction"""
        injected_documents = "```\n" + "\n".join([
            PromptTemplates._format_chunk(result)
            for result in retrieved_chunks
        ]) + "\n```\n"
        
        return f"""# INFORMATION EXTRACTION

            ## RETRIEVED CHUNKS
            {injected_documents}

            ## SEARCH QUERY
            Extract all occurrences of: {query}

            ## INSTRUCTIONS
            1. Locate **ALL** occurrences related to the search query
            2. For each occurrence, provide:
            - Chunk ID: [chunk_id]
            - Document: document name
            - Context: relevant excerpt
            - Relevance score

            3. If no information is found, clearly state this

            Extract now:"""
    
    @staticmethod
    def question_answer_template(
        question: str,
        retrieved_chunks: List[Dict]
    ) -> str:
        """Template optimized for question-answering"""
        injected_documents = "```\n" + "\n".join([
            PromptTemplates._format_chunk(result)
            for result in retrieved_chunks
        ]) + "\n```\n"
        
        return f"""# QUESTION-ANSWERING SYSTEM

            ## RETRIEVED CONTEXT
            {injected_documents}

            ## QUESTION
            {question}

            ## INSTRUCTIONS
            1. Answer in a **direct and objective** manner
            2. Base your answer **exclusively** on the provided chunks
            3. Cite sources using [chunk_id]
            4. If the answer is in multiple chunks, **synthesize** the information
            5. If there's insufficient information, clearly state this

            Answer now:"""
    
    @staticmethod
    def fact_verification_template(
        claim: str,
        retrieved_chunks: List[Dict]
    ) -> str:
        """Template for fact verification and validation"""
        injected_documents = "```\n" + "\n".join([
            PromptTemplates._format_chunk(result)
            for result in retrieved_chunks
        ]) + "\n```\n"
        
        return f"""# FACT VERIFICATION SYSTEM

            ## RETRIEVED EVIDENCE
            {injected_documents}

            ## CLAIM TO VERIFY
            "{claim}"

            ## INSTRUCTIONS
            1. Analyze the claim against the retrieved evidence
            2. Provide one of these verdicts:
            - SUPPORTED: Strong evidence supports the claim [cite sources]
            - PARTIALLY SUPPORTED: Some evidence supports but with limitations [cite sources]
            - CONTRADICTED: Evidence contradicts the claim [cite sources]
            - INSUFFICIENT_EVIDENCE: Not enough information to verify

            3. For each verdict, provide specific citations and reasoning

            Verify now:"""
    
    @staticmethod
    def technical_explanation_template(
        concept: str,
        retrieved_chunks: List[Dict]
    ) -> str:
        """Template for technical explanations and definitions"""
        injected_documents = "```\n" + "\n".join([
            PromptTemplates._format_chunk(result)
            for result in retrieved_chunks
        ]) + "\n```\n"
        
        return f"""# TECHNICAL EXPLANATION SYSTEM

            ## TECHNICAL CONTEXT
            {injected_documents}

            ## CONCEPT TO EXPLAIN
            {concept}

            ## INSTRUCTIONS
            1. Provide a clear technical explanation
            2. Include:
            - Definition and purpose [cite: chunk_id]
            - Key characteristics and components [cite: chunk_id]
            - Practical applications or examples [cite: chunk_id]
            - Related concepts or alternatives [cite: chunk_id]

            3. Use precise technical language while maintaining clarity
            4. Cite specific chunks for each piece of information

            Explain now:"""

    @staticmethod
    def get_available_templates() -> Dict[str, str]:
        """Returns available template types and their descriptions"""
        return {
            "query": "Advanced query with precise citations",
            "summary": "Structured document summarization", 
            "comparison": "Comparative analysis between documents",
            "extraction": "Specific information extraction",
            "qa": "Optimized question-answering",
            "verification": "Fact checking and validation",
            "technical": "Technical explanations and definitions"
        }