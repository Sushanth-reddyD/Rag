"""Text chunking strategies for document segmentation."""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    start_offset: int
    end_offset: int
    chunk_index: int
    metadata: Dict[str, Any]


class ChunkingStrategy:
    """Base class for chunking strategies."""
    
    def chunk(self, text: str, document_metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Split text into chunks.
        
        Args:
            text: Text to chunk
            document_metadata: Metadata from document
            
        Returns:
            List of Chunk objects
        """
        raise NotImplementedError("Subclasses must implement chunk()")


class FixedSizeChunker(ChunkingStrategy):
    """Chunks text into fixed-size overlapping windows."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        separator: str = "\n"
    ):
        """
        Initialize fixed-size chunker.
        
        Args:
            chunk_size: Target size of each chunk in characters
            overlap: Number of overlapping characters between chunks
            separator: Preferred separator for chunk boundaries
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separator = separator
    
    def chunk(self, text: str, document_metadata: Dict[str, Any]) -> List[Chunk]:
        """Split text into fixed-size chunks with overlap."""
        chunks = []
        text_length = len(text)
        chunk_index = 0
        current_pos = 0
        
        while current_pos < text_length:
            # Calculate chunk end position
            end_pos = min(current_pos + self.chunk_size, text_length)
            
            # Try to find a good break point (separator) near the end
            if end_pos < text_length:
                # Look for separator within last 20% of chunk
                search_start = end_pos - int(self.chunk_size * 0.2)
                chunk_text = text[current_pos:end_pos]
                
                # Find last occurrence of separator
                last_sep = chunk_text.rfind(self.separator, search_start - current_pos)
                if last_sep != -1:
                    end_pos = current_pos + last_sep + len(self.separator)
            
            # Extract chunk text
            chunk_text = text[current_pos:end_pos].strip()
            
            if chunk_text:  # Only add non-empty chunks
                chunk = Chunk(
                    text=chunk_text,
                    start_offset=current_pos,
                    end_offset=end_pos,
                    chunk_index=chunk_index,
                    metadata={
                        'document_id': document_metadata.get('document_id'),
                        'overlap_size': self.overlap if chunk_index > 0 else 0,
                        'chunking_method': 'fixed_size'
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move to next chunk with overlap
            current_pos = end_pos - self.overlap
            
            # Prevent infinite loop if overlap >= chunk_size
            if current_pos <= chunks[-1].start_offset if chunks else False:
                current_pos = end_pos
        
        return chunks


class SentenceChunker(ChunkingStrategy):
    """Chunks text based on sentence boundaries."""
    
    def __init__(
        self,
        target_chunk_size: int = 512,
        max_chunk_size: int = 768,
        overlap_sentences: int = 1
    ):
        """
        Initialize sentence-based chunker.
        
        Args:
            target_chunk_size: Target size in characters
            max_chunk_size: Maximum chunk size before forcing split
            overlap_sentences: Number of sentences to overlap
        """
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences
        
        # Simple sentence boundary pattern
        self.sentence_pattern = re.compile(r'[.!?]+\s+')
    
    def _split_into_sentences(self, text: str) -> List[Tuple[str, int]]:
        """Split text into sentences with their positions."""
        sentences = []
        last_end = 0
        
        for match in self.sentence_pattern.finditer(text):
            sentence = text[last_end:match.end()].strip()
            if sentence:
                sentences.append((sentence, last_end))
            last_end = match.end()
        
        # Add final sentence if exists
        if last_end < len(text):
            final_sentence = text[last_end:].strip()
            if final_sentence:
                sentences.append((final_sentence, last_end))
        
        return sentences
    
    def chunk(self, text: str, document_metadata: Dict[str, Any]) -> List[Chunk]:
        """Split text into chunks based on sentence boundaries."""
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        chunks = []
        chunk_index = 0
        current_chunk_sentences = []
        current_chunk_size = 0
        start_offset = sentences[0][1] if sentences else 0
        
        for i, (sentence, offset) in enumerate(sentences):
            sentence_size = len(sentence)
            
            # Check if adding this sentence would exceed max size
            if (current_chunk_size + sentence_size > self.max_chunk_size and 
                current_chunk_sentences):
                # Create chunk from accumulated sentences
                chunk_text = ' '.join(current_chunk_sentences)
                chunk = Chunk(
                    text=chunk_text,
                    start_offset=start_offset,
                    end_offset=start_offset + len(chunk_text),
                    chunk_index=chunk_index,
                    metadata={
                        'document_id': document_metadata.get('document_id'),
                        'num_sentences': len(current_chunk_sentences),
                        'chunking_method': 'sentence_based'
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk_sentences) - self.overlap_sentences)
                current_chunk_sentences = current_chunk_sentences[overlap_start:]
                start_offset = sentences[i - len(current_chunk_sentences)][1] if current_chunk_sentences else offset
                current_chunk_size = sum(len(s) for s in current_chunk_sentences)
            
            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_chunk_size += sentence_size
            
            # Check if we've reached target size
            if current_chunk_size >= self.target_chunk_size:
                chunk_text = ' '.join(current_chunk_sentences)
                chunk = Chunk(
                    text=chunk_text,
                    start_offset=start_offset,
                    end_offset=start_offset + len(chunk_text),
                    chunk_index=chunk_index,
                    metadata={
                        'document_id': document_metadata.get('document_id'),
                        'num_sentences': len(current_chunk_sentences),
                        'chunking_method': 'sentence_based'
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Prepare for next chunk with overlap
                overlap_start = max(0, len(current_chunk_sentences) - self.overlap_sentences)
                current_chunk_sentences = current_chunk_sentences[overlap_start:]
                start_offset = sentences[min(i + 1, len(sentences) - 1)][1]
                current_chunk_size = sum(len(s) for s in current_chunk_sentences)
        
        # Add remaining sentences as final chunk
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunk = Chunk(
                text=chunk_text,
                start_offset=start_offset,
                end_offset=start_offset + len(chunk_text),
                chunk_index=chunk_index,
                metadata={
                    'document_id': document_metadata.get('document_id'),
                    'num_sentences': len(current_chunk_sentences),
                    'chunking_method': 'sentence_based'
                }
            )
            chunks.append(chunk)
        
        return chunks


class SemanticChunker(ChunkingStrategy):
    """
    Semantic chunking based on structure (sections, paragraphs).
    Falls back to sentence chunking if no structure available.
    """
    
    def __init__(
        self,
        target_chunk_size: int = 512,
        max_chunk_size: int = 1024
    ):
        """
        Initialize semantic chunker.
        
        Args:
            target_chunk_size: Target chunk size in characters
            max_chunk_size: Maximum chunk size
        """
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.sentence_chunker = SentenceChunker(
            target_chunk_size=target_chunk_size,
            max_chunk_size=max_chunk_size
        )
    
    def chunk(self, text: str, document_metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk text based on semantic structure."""
        sections = document_metadata.get('sections', [])
        
        # If no structural information, fall back to sentence chunking
        if not sections:
            return self.sentence_chunker.chunk(text, document_metadata)
        
        chunks = []
        chunk_index = 0
        
        for section in sections:
            section_type = section.get('type', 'unknown')
            start = section.get('start_offset', 0)
            end = section.get('end_offset', len(text))
            
            section_text = text[start:end].strip()
            
            if not section_text:
                continue
            
            # If section is small enough, keep it as one chunk
            if len(section_text) <= self.max_chunk_size:
                chunk = Chunk(
                    text=section_text,
                    start_offset=start,
                    end_offset=end,
                    chunk_index=chunk_index,
                    metadata={
                        'document_id': document_metadata.get('document_id'),
                        'section_type': section_type,
                        'section_header': section.get('text', ''),
                        'chunking_method': 'semantic_structure'
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Split large sections using sentence chunker
                section_chunks = self.sentence_chunker.chunk(
                    section_text,
                    document_metadata
                )
                
                for sub_chunk in section_chunks:
                    chunk = Chunk(
                        text=sub_chunk.text,
                        start_offset=start + sub_chunk.start_offset,
                        end_offset=start + sub_chunk.end_offset,
                        chunk_index=chunk_index,
                        metadata={
                            **sub_chunk.metadata,
                            'section_type': section_type,
                            'section_header': section.get('text', ''),
                            'chunking_method': 'semantic_structure_with_sentence_split'
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1
        
        return chunks


class DocumentChunker:
    """Main chunker that orchestrates different chunking strategies."""
    
    def __init__(
        self,
        strategy: str = "semantic",
        chunk_size: int = 512,
        overlap: int = 50,
        max_chunk_size: int = 1024
    ):
        """
        Initialize document chunker.
        
        Args:
            strategy: Chunking strategy ('fixed', 'sentence', 'semantic')
            chunk_size: Target chunk size
            overlap: Overlap size for fixed strategy
            max_chunk_size: Maximum chunk size
        """
        self.strategy = strategy
        
        if strategy == "fixed":
            self.chunker = FixedSizeChunker(
                chunk_size=chunk_size,
                overlap=overlap
            )
        elif strategy == "sentence":
            self.chunker = SentenceChunker(
                target_chunk_size=chunk_size,
                max_chunk_size=max_chunk_size
            )
        elif strategy == "semantic":
            self.chunker = SemanticChunker(
                target_chunk_size=chunk_size,
                max_chunk_size=max_chunk_size
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def chunk_document(
        self,
        text: str,
        document_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Chunk document text using configured strategy.
        
        Args:
            text: Document text to chunk
            document_metadata: Document metadata
            
        Returns:
            List of chunks
        """
        return self.chunker.chunk(text, document_metadata)
