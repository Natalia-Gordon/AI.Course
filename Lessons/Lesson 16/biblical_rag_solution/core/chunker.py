"""
Text chunking functionality for Biblical texts.
Aggregates verses into larger chunks (chapters or multi-verse groups) for better retrieval context.
"""
import logging
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Tuple

from ..config import ChunkConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TextChunker:
    """Handles text chunking for Biblical texts."""
    
    def __init__(self, chunk_config: ChunkConfig):
        self.config = chunk_config
        
    def chunk_documents(self, documents: Iterator[Tuple[str, str, Dict[str, Any]]]) -> Iterator[Dict[str, Any]]:
        """
        Chunk documents according to the configuration.
        
        Args:
            documents: Iterator of (category, filename, record) tuples
            
        Yields:
            Dict[str, Any]: Chunked document with aggregated text and metadata
        """
        if self.config.chunk_by == "verse":
            # No chunking - return individual verses
            for category, filename, record in documents:
                chunk = self._create_verse_chunk(category, filename, record)
                if chunk is not None:
                    yield chunk
                
        elif self.config.chunk_by == "chapter":
            # Chunk by chapters
            yield from self._chunk_by_chapter(documents)
            
        elif self.config.chunk_by == "multi_verse":
            # Chunk by multiple verses
            yield from self._chunk_by_multi_verse(documents)
            
        elif self.config.chunk_by == "balanced":
            # Balanced chunking: Tanach by chapters, non-Tanach by files
            yield from self._chunk_by_balanced(documents)
            
        else:
            logging.warning(f"Unknown chunk_by value: {self.config.chunk_by}. Defaulting to verse-level.")
            for category, filename, record in documents:
                chunk = self._create_verse_chunk(category, filename, record)
                if chunk is not None:
                    yield chunk
    
    def _create_verse_chunk(self, category: str, filename: str, record: Dict[str, Any]) -> Dict[str, Any]:
        """Create a chunk from a single verse."""
        # Get the text content (Hebrew for Tanach, text for others)
        if category.lower() == 'tanach' or category.lower() == 'tanach_backup_with_pisuk':
            text_content = record.get('text_he', '').strip()
            # For Tanach: use both chapter and verse
            chapter_info = record.get('chapter')
            verse_num = record.get('verse')
        else:
            text_content = record.get('text', '').strip()
            # For Bavli/Mishnah/etc: use section as both chapter and verse
            chapter_info = record.get('section')
            verse_num = record.get('section')
        
        # Only create chunk if we have actual text content
        if not text_content:
            return None
        
        return {
            'text': text_content,
            'category': category,
            'book': record.get('book', filename.replace('.jsonl', '')),
            'chapter': chapter_info,
            'verse': verse_num,
            'verses': [verse_num] if verse_num is not None else [],
            'chunk_type': 'verse',
            'chunk_size': 1,
            'source_file': filename
        }
    
    def _chunk_by_chapter(self, documents: Iterator[Tuple[str, str, Dict[str, Any]]]) -> Iterator[Dict[str, Any]]:
        """Chunk documents by chapter - combine all verses in a chapter into one text."""
        chapters = defaultdict(lambda: {
            'verses': [],
            'texts': [],
            'category': '',
            'book': '',
            'chapter': None,
            'source_file': ''
        })
        
        # Group verses by chapter
        for category, filename, record in documents:
            book = record.get('book', filename.replace('.jsonl', ''))
            
            # For Tanach: use chapter field, for others: treat each record as individual chunk
            if category.lower() == 'tanach' or category.lower() == 'tanach_backup_with_pisuk':
                chapter = record.get('chapter')
                if chapter is None:
                    # If no chapter info, treat as individual verse
                    chunk = self._create_verse_chunk(category, filename, record)
                    if chunk is not None:
                        yield chunk
                    continue
            else:
                # For Bavli/Mishnah/Tosefta/Yerushalmi: each section is already a complete unit
                # Don't group by chapter, just return individual sections
                chunk = self._create_verse_chunk(category, filename, record)
                if chunk is not None:
                    yield chunk
                continue
                
            # Only Tanach data gets here - group by actual chapters
            chapter_key = f"{category}_{book}_{chapter}"
            chapter_data = chapters[chapter_key]
            
            chapter_data['category'] = category
            chapter_data['book'] = book
            chapter_data['chapter'] = chapter
            chapter_data['source_file'] = filename
            
            # For Tanach, use verse field
            verse_num = record.get('verse')
            chapter_data['verses'].append(verse_num)
            
            # Get the Hebrew text for Tanach
            text_content = record.get('text_he', '').strip()
            if text_content:
                chapter_data['texts'].append(text_content)
        
        # Yield completed chapters
        for chapter_key, chapter_data in chapters.items():
            if chapter_data['texts']:  # Only if we have actual text content
                combined_text = ' '.join(chapter_data['texts'])
                
                # Handle verse range properly - filter out None values
                valid_verses = [v for v in chapter_data['verses'] if v is not None]
                if valid_verses:
                    verse_range = f"{min(valid_verses)}-{max(valid_verses)}" if len(valid_verses) > 1 else str(valid_verses[0])
                else:
                    verse_range = None
                
                yield {
                    'text': combined_text,
                    'category': chapter_data['category'],
                    'book': chapter_data['book'],
                    'chapter': chapter_data['chapter'],
                    'verse': verse_range,
                    'verses': sorted(valid_verses),
                    'chunk_type': 'chapter',
                    'chunk_size': len(chapter_data['verses']),
                    'source_file': chapter_data['source_file']
                }
    
    def _chunk_by_multi_verse(self, documents: Iterator[Tuple[str, str, Dict[str, Any]]]) -> Iterator[Dict[str, Any]]:
        """Chunk documents by multiple verses with optional overlap."""
        books = defaultdict(lambda: {
            'verses': [],
            'category': '',
            'book': '',
            'source_file': ''
        })
        
        # Group verses by book
        for category, filename, record in documents:
            book = record.get('book', filename.replace('.jsonl', ''))
            book_key = f"{category}_{book}"
            
            books[book_key]['category'] = category
            books[book_key]['book'] = book
            books[book_key]['source_file'] = filename
            books[book_key]['verses'].append(record)
        
        # Create multi-verse chunks for each book
        for book_key, book_data in books.items():
            verses = book_data['verses']
            
            # Sort verses by chapter/section and verse number
            verses.sort(key=lambda v: (v.get('chapter') or v.get('section', 0), v.get('verse', 0)))
            
            # Create chunks with sliding window
            chunk_size = self.config.verses_per_chunk
            overlap = self.config.chunk_overlap
            step = max(1, chunk_size - overlap)
            
            for i in range(0, len(verses), step):
                chunk_verses = verses[i:i + chunk_size]
                
                if not chunk_verses:
                    continue
                
                # Get text from verses
                category = book_data['category']
                texts = []
                for verse in chunk_verses:
                    if category.lower() == 'tanach' or category.lower() == 'tanach_backup_with_pisuk':
                        text_content = verse.get('text_he', '').strip()
                    else:
                        text_content = verse.get('text', '').strip()
                    
                    if text_content:
                        texts.append(text_content)
                
                if not texts:
                    continue
                
                # Get verse/section numbers - handle both verse and section fields
                verse_numbers = []
                for v in chunk_verses:
                    verse_num = v.get('verse') or v.get('section')
                    if verse_num is not None:
                        verse_numbers.append(verse_num)
                
                chapters = [v.get('chapter') or v.get('section') for v in chunk_verses]
                
                # Create verse range string
                if verse_numbers:
                    first_verse = verse_numbers[0]
                    last_verse = verse_numbers[-1]
                    verse_range = f"{first_verse}-{last_verse}" if first_verse != last_verse else str(first_verse)
                else:
                    verse_range = None
                
                # Get chapter range
                first_chapter = chapters[0] if chapters else None
                last_chapter = chapters[-1] if chapters else None
                chapter_info = first_chapter if first_chapter == last_chapter else f"{first_chapter}-{last_chapter}"
                
                yield {
                    'text': ' '.join(texts),
                    'category': book_data['category'],
                    'book': book_data['book'],
                    'chapter': chapter_info,
                    'verse': verse_range,
                    'verses': verse_numbers,
                    'chunk_type': 'multi_verse',
                    'chunk_size': len(chunk_verses),
                    'source_file': book_data['source_file']
                }

    def _chunk_by_balanced(self, documents: Iterator[Tuple[str, str, Dict[str, Any]]]) -> Iterator[Dict[str, Any]]:
        """
        Balanced chunking: Tanach by chapters, non-Tanach by entire files.
        This provides better representation balance between source types.
        """
        tanach_chapters = defaultdict(lambda: {
            'verses': [],
            'texts': [],
            'category': '',
            'book': '',
            'chapter': None,
            'source_file': ''
        })
        
        # Storage for file-level chunks (non-Tanach)
        file_chunks = defaultdict(lambda: {
            'sections': [],
            'texts': [],
            'category': '',
            'book': '',
            'source_file': ''
        })
        
        # Group data appropriately
        for category, filename, record in documents:
            book = record.get('book', filename.replace('.jsonl', ''))
            
            # For Tanach: group by chapters (same as existing chapter logic)
            if category.lower() == 'tanach' or category.lower() == 'tanach_backup_with_pisuk':
                chapter = record.get('chapter')
                if chapter is None:
                    # If no chapter info, treat as individual verse
                    chunk = self._create_verse_chunk(category, filename, record)
                    if chunk is not None:
                        yield chunk
                    continue
                
                # Group Tanach by chapters
                chapter_key = f"{category}_{book}_{chapter}"
                chapter_data = tanach_chapters[chapter_key]
                
                chapter_data['category'] = category
                chapter_data['book'] = book
                chapter_data['chapter'] = chapter
                chapter_data['source_file'] = filename
                
                verse_num = record.get('verse')
                chapter_data['verses'].append(verse_num)
                
                # Get Hebrew text for Tanach
                text_content = record.get('text_he', '').strip()
                if text_content:
                    chapter_data['texts'].append(text_content)
            else:
                # For non-Tanach: group by entire file
                file_key = f"{category}_{filename}"
                file_data = file_chunks[file_key]
                
                file_data['category'] = category
                file_data['book'] = book
                file_data['source_file'] = filename
                
                # Collect all sections from this file
                section_num = record.get('section')
                if section_num is not None:
                    file_data['sections'].append(section_num)
                
                # Get text content
                text_content = record.get('text', '').strip()
                if text_content:
                    file_data['texts'].append(text_content)
        
        # Yield Tanach chapters (same as existing chapter chunking)
        for chapter_key, chapter_data in tanach_chapters.items():
            if chapter_data['texts']:  # Only if we have actual text content
                combined_text = ' '.join(chapter_data['texts'])
                
                # Handle verse range properly
                valid_verses = [v for v in chapter_data['verses'] if v is not None]
                if valid_verses:
                    verse_range = f"{min(valid_verses)}-{max(valid_verses)}" if len(valid_verses) > 1 else str(valid_verses[0])
                else:
                    verse_range = None
                
                yield {
                    'text': combined_text,
                    'category': chapter_data['category'],
                    'book': chapter_data['book'],
                    'chapter': chapter_data['chapter'],
                    'verse': verse_range,
                    'verses': sorted(valid_verses),
                    'chunk_type': 'chapter',
                    'chunk_size': len(chapter_data['verses']),
                    'source_file': chapter_data['source_file']
                }
        
        # Yield non-Tanach files as single chunks
        for file_key, file_data in file_chunks.items():
            if file_data['texts']:  # Only if we have actual text content
                combined_text = ' '.join(file_data['texts'])
                
                # Handle section range
                valid_sections = [s for s in file_data['sections'] if s is not None]
                if valid_sections:
                    section_range = f"{min(valid_sections)}-{max(valid_sections)}" if len(valid_sections) > 1 else str(valid_sections[0])
                else:
                    section_range = None
                
                yield {
                    'text': combined_text,
                    'category': file_data['category'],
                    'book': file_data['book'],
                    'chapter': file_data['book'],  # Use book name as "chapter" for non-Tanach
                    'verse': section_range,
                    'verses': sorted(valid_sections) if valid_sections else [],
                    'chunk_type': 'file',  # New chunk type for file-level chunks
                    'chunk_size': len(file_data['texts']),  # Number of original records
                    'source_file': file_data['source_file']
                }


def create_chunked_corpus(data_dir: str, chunk_config: ChunkConfig) -> Iterator[Dict[str, Any]]:
    """
    Create a chunked corpus from the data directory.
    
    Args:
        data_dir: Path to the data directory
        chunk_config: Configuration for chunking
        
    Yields:
        Dict[str, Any]: Chunked documents
    """
    from .data_loader import load_all_data_generator
    
    logging.info(f"Creating chunked corpus with strategy: {chunk_config.chunk_by}")
    if chunk_config.chunk_by == "multi_verse":
        logging.info(f"Multi-verse settings: {chunk_config.verses_per_chunk} verses per chunk, {chunk_config.chunk_overlap} overlap")
    elif chunk_config.chunk_by == "balanced":
        logging.info("Balanced chunking: Tanach by chapters, non-Tanach by complete files for optimal retrieval balance")
    
    chunker = TextChunker(chunk_config)
    documents = load_all_data_generator(data_dir)
    
    chunk_count = 0
    for chunk in chunker.chunk_documents(documents):
        chunk_count += 1
        if chunk_count % 100 == 0:
            logging.info(f"Processed {chunk_count} chunks...")
        yield chunk
    
    logging.info(f"Finished creating chunked corpus. Total chunks: {chunk_count}")
