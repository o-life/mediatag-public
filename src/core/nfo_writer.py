"""NFO file generator for video metadata."""
import json
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import Optional, List


def _extract_plot_text(ai_raw_text: Optional[str]) -> str:
    """Extract readable plot text from AI raw output.

    Strips JSON/markdown formatting and returns clean text.
    """
    if not ai_raw_text:
        return ""

    text = ai_raw_text.strip()

    # Strip markdown code block wrappers
    code_block_match = re.match(r'^```(?:json)?\s*\n?(.*?)\n?\s*```$', text, re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1).strip()

    # Try to parse as JSON and extract meaningful values
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            parts = []
            for key, value in data.items():
                if isinstance(value, list) and value:
                    clean_key = key.replace('/', '').replace(' ', '')
                    parts.append(f"{clean_key}: {', '.join(str(v) for v in value if v)}")
                elif isinstance(value, str) and value:
                    parts.append(f"{key}: {value}")
            return ' | '.join(parts) if parts else text
        elif isinstance(data, list):
            return ', '.join(str(item) for item in data if item)
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: return raw text cleaned
    return text


def generate_nfo(
    video_path: str,
    mapped_tags: list,
    ai_raw_text: Optional[str] = None,
    collection_name: Optional[str] = None,
    actor_name: Optional[str] = None,
    actor_aliases: Optional[List[str]] = None
) -> str:
    """
    Generate NFO file for video with mapped tags and actor information.

    Args:
        video_path: Path to video file
        mapped_tags: List of standard tags from dictionary
        ai_raw_text: Raw AI analysis text (optional, for reference)
        collection_name: Collection/set name for Jellyfin (optional)
        actor_name: Primary actor name (optional)
        actor_aliases: List of actor aliases (optional)

    Returns:
        Path to generated NFO file
    """
    video_file = Path(video_path)
    nfo_path = video_file.with_suffix('.nfo')

    # Create XML structure
    root = ET.Element('movie')

    # Title (use filename without extension)
    title = ET.SubElement(root, 'title')
    title.text = video_file.stem

    # Tags/Genres (must come before actor for Jellyfin compatibility)
    for tag in mapped_tags:
        genre = ET.SubElement(root, 'genre')
        genre.text = tag

    # Actor information (for Jellyfin/Emby)
    if actor_name:
        actor_elem = ET.SubElement(root, 'actor')

        name_elem = ET.SubElement(actor_elem, 'name')
        name_elem.text = actor_name

        # Add aliases as alternative names
        if actor_aliases:
            for alias in actor_aliases:
                alias_elem = ET.SubElement(actor_elem, 'alias')
                alias_elem.text = alias

    # Plot (extract readable text from AI output, not raw JSON)
    if ai_raw_text:
        plot_text = _extract_plot_text(ai_raw_text)
        if plot_text:
            plot = ET.SubElement(root, 'plot')
            plot.text = plot_text

    # Tags (for Infuse/VidHub compatibility)
    for tag in mapped_tags:
        tag_elem = ET.SubElement(root, 'tag')
        tag_elem.text = tag

    # Collection/Set (for Jellyfin)
    if collection_name:
        set_elem = ET.SubElement(root, 'set')
        set_elem.text = collection_name

    # Pretty print XML
    xml_str = ET.tostring(root, encoding='unicode')
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent='  ')

    # Remove extra blank lines
    lines = [line for line in pretty_xml.split('\n') if line.strip()]
    pretty_xml = '\n'.join(lines)

    # Write to file
    with open(nfo_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)

    return str(nfo_path)
