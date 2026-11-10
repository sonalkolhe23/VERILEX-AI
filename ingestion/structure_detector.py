import re


class StructureDetector:
    """Detects chapters, sections and articles in legal PDF text."""
    
    
    CHAPTER_PATTERN = re.compile(
        r"^\s*CHAPTER\s+([IVXLCDM]+)\s*$",
        re.IGNORECASE
    )
    
    SECTION_PATTERN = re.compile(
        r"^\s*SECTION\s+(\d+)\s*$",
        re.IGNORECASE
    )
    
    ARTICLE_PATTERN = re.compile(
        r"^\s*Article\s+(\d+)\s*$",
        re.IGNORECASE
    )
    
    def detect(self, pages):
        current_chapter = None
        current_section = None
        current_article = None
        
        structured_pages = []
        all_headings = []
        
        
        for page in pages:
            page_articles = []
            page_headings = []
            
            
            
            for line in page["text"].splitlines():
                line = line.strip()
                
                if not line:
                    continue
                
                chapter_match =  self.CHAPTER_PATTERN.match(line)
                section_match = self.SECTION_PATTERN.match(line)
                article_match = self.ARTICLE_PATTERN.match(line)
                
                if chapter_match:
                    current_chapter = (
                        f"Chapter {chapter_match.group(1)}"
                    )
                    current_section = None
                    current_article = None
                    
                    heading = {
                        "type": "chapter",
                        "heading": current_chapter,
                        "number": chapter_match.group(1).upper(),
                        "page_number": page["page_number"]
                    }
                
                elif section_match:
                    current_section = (
                        f"Section {section_match.group(1)}"
                    )
                    current_article = None
                    
                    heading = {
                        "type": "section",
                        "heading": current_section,
                        "number": section_match.group(1),
                        "page_number": page["page_number"]
                    }

                elif article_match:
                    current_article = (
                        f"Article {article_match.group(1)}"
                    )
                    
                    article_heading = {
                        "type": "article",
                        "heading": current_article,
                        "number": article_match.group(1),
                        "page_number": page["page_number"],
                        "chapter": current_chapter,
                        "section": current_section
                    }
                    
                    page_articles.append(article_heading)
                    heading = article_heading
                else:
                    continue
                
                page_headings.append(heading)
                all_headings.append(heading)
                
            structured_pages.append({
                **page,
                "chapter": current_chapter,
                "section": current_section,
                "article": current_article,
                "articles_on_page": page_articles,
                "headings" : page_headings
            })
            
        return {
            "pages": structured_pages,
            "headings": all_headings
        }