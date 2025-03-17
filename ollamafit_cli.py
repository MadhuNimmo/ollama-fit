#!/usr/bin/env python3
"""
Ollama Model Library Scraper - Fetches model information from the Ollama website
"""


import requests
from bs4 import BeautifulSoup
import json
import re
import os
import sys
import time
import platform
import subprocess
import psutil  # Add this line
from typing import Dict, List, Any, Optional

class OllamaLibraryScraper:
    def __init__(self, cache_dir=".ollama_cache"):
        """Initialize the scraper with optional cache directory."""
        self.base_url = "https://ollama.com"
        self.library_url = f"{self.base_url}/library"
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        # Initialize empty model data
        self.models = []
        
    def scrape_library(self, use_cache=True, cache_expiry=3600):
        """
        Scrape the main library page to get a list of all models.
        
        Args:
            use_cache: Whether to use cached results if available
            cache_expiry: How long (in seconds) to consider cache valid
        """
        cache_file = os.path.join(self.cache_dir, "library.html")
        
        # Check if cache exists and is recent enough
        if use_cache and os.path.exists(cache_file):
            cache_age = time.time() - os.path.getmtime(cache_file)
            if cache_age < cache_expiry:
                print(f"Using cached library page (age: {int(cache_age)}s)")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    html = f.read()
            else:
                print("Cache expired, fetching fresh data...")
                html = self._fetch_page(self.library_url)
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(html)
        else:
            print("Fetching library page...")
            html = self._fetch_page(self.library_url)
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(html)
                
        # Parse the HTML to extract model cards
        soup = BeautifulSoup(html, 'html.parser')
        model_cards = soup.select('a[href^="/library/"]')
        
        # Process each model card
        for card in model_cards:
            # Skip tags pages
            if "/tags" in card['href']:
                continue
                
            # Get model name from URL
            model_name = card['href'].split('/')[-1]
            
            # Skip if this is not a model card (might be navigation, etc.)
            if not model_name or len(model_name) < 2:
                continue
                
            # Extract basic information from the card
            model_info = {
                'name': model_name,
                'url': f"{self.base_url}{card['href']}",
                'description': self._extract_description(card),
                'pull_count': self._extract_pull_count(card),
                'last_updated': self._extract_last_updated(card),
                'capabilities': self._extract_capabilities(card),
                'size': self._extract_size(card)
            }
            
            self.models.append(model_info)
            
        print(f"Found {len(self.models)} models")
        return self.models
    
    def _extract_description(self, card):
        """Extract model description from card."""
        desc_elem = card.select_one('p.max-w-lg')
        return desc_elem.text.strip() if desc_elem else ""
    
    def _extract_pull_count(self, card):
        """Extract pull count from card."""
        pull_elem = card.select_one('span[x-test-pull-count]')
        return pull_elem.text.strip() if pull_elem else "Unknown"
    
    def _extract_last_updated(self, card):
        """Extract last updated date from card."""
        updated_elem = card.select_one('span[x-test-updated]')
        return updated_elem.text.strip() if updated_elem else "Unknown"
    
    def _extract_capabilities(self, card):
        """Extract model capabilities from card."""
        capabilities = []
        capability_elems = card.select('span[x-test-capability]')
        for elem in capability_elems:
            capabilities.append(elem.text.strip())
        return capabilities
    
    def _extract_size(self, card):
        """Extract model size from card."""
        size_elem = card.select_one('span[x-test-size]')
        return size_elem.text.strip() if size_elem else "Unknown"
    
    def scrape_model_details(self, model_info, use_cache=True, cache_expiry=3600):
        """
        Scrape detailed information about a specific model.
        
        Args:
            model_info: Basic model information from scrape_library
            use_cache: Whether to use cached results if available
            cache_expiry: How long (in seconds) to consider cache valid
        """
        model_name = model_info['name']
        model_url = model_info['url']
        cache_file = os.path.join(self.cache_dir, f"model_{model_name}.html")
        
        # Check if cache exists and is recent enough
        if use_cache and os.path.exists(cache_file):
            cache_age = time.time() - os.path.getmtime(cache_file)
            if cache_age < cache_expiry:
                print(f"Using cached model page for {model_name} (age: {int(cache_age)}s)")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    html = f.read()
            else:
                print(f"Cache expired for {model_name}, fetching fresh data...")
                html = self._fetch_page(model_url)
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(html)
        else:
            print(f"Fetching details for {model_name}...")
            html = self._fetch_page(model_url)
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(html)
                
        # Parse the HTML to extract detailed information
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract README content
        readme = self._extract_readme(soup)
        
        # Extract parameter count if available
        parameter_count = self._extract_parameter_count(soup, readme)
        
        # Extract context window if available
        context_window = self._extract_context_window(soup, readme)
        
        # Update model_info with details
        model_info.update({
            'readme': readme,
            'parameter_count': parameter_count,
            'context_window': context_window
        })
        
        # Scrape tags for this model
        self.scrape_model_tags(model_info, use_cache, cache_expiry)
        
        return model_info
    
    def scrape_model_tags(self, model_info, use_cache=True, cache_expiry=3600):
        """Scrape all available tags for a specific model."""
        model_name = model_info['name']
        tags_url = f"{self.base_url}/library/{model_name}/tags"
        cache_file = os.path.join(self.cache_dir, f"tags_{model_name}.html")
        
        # Check if cache exists and is recent enough
        if use_cache and os.path.exists(cache_file):
            cache_age = time.time() - os.path.getmtime(cache_file)
            if cache_age < cache_expiry:
                #print(f"Using cached tags for {model_name} (age: {int(cache_age)}s)")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    html = f.read()
            else:
                #print(f"Cache expired for {model_name} tags, fetching fresh data...")
                html = self._fetch_page(tags_url)
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(html)
        else:
            #print(f"Fetching tags for {model_name}...")
            html = self._fetch_page(tags_url)
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(html)
                
        # Parse the HTML to extract tags
        soup = BeautifulSoup(html, 'html.parser')
        tag_links = soup.select('a.group')
        
        tags = []
        for tag_link in tag_links:
            # Extract tag name from URL
            tag_name = tag_link['href'].split('/')[-1]
            
            # Find the size information
            size_info = tag_link.find_next('span', class_='text-neutral-500')
            size_text = size_info.text.strip() if size_info else "Unknown"
            
            # Extract size in bytes if possible
            size_bytes = None
            size_match = re.search(r'(\d+(?:\.\d+)?)\s*(KB|MB|GB|TB)', size_text)
            if size_match:
                size_value = float(size_match.group(1))
                size_unit = size_match.group(2)
                
                # Convert to bytes
                unit_multipliers = {
                    'KB': 1024,
                    'MB': 1024**2,
                    'GB': 1024**3,
                    'TB': 1024**4
                }
                
                size_bytes = size_value * unit_multipliers.get(size_unit, 1)
            
            # Find last updated info
            updated_info = size_info.find_next('span', class_='text-neutral-500') if size_info else None
            updated_text = updated_info.text.strip() if updated_info else "Unknown"
            
            # Extract quantization information from tag name
            quantization = None
            quant_patterns = {
                'q2': '2-bit',
                'q3': '3-bit',
                'q4': '4-bit', 
                'q5': '5-bit',
                'q6': '6-bit',
                'q8': '8-bit'
            }
            
            for pattern, quant_type in quant_patterns.items():
                if pattern in tag_name.lower():
                    quantization = quant_type
                    break
                    
            if not quantization and 'int4' in tag_name.lower():
                quantization = '4-bit'
            elif not quantization and 'int8' in tag_name.lower():
                quantization = '8-bit'
            elif not quantization and 'fp16' in tag_name.lower():
                quantization = '16-bit'
            elif not quantization:
                quantization = '16-bit'  # Default assumption
            
            # Extract parameters from tag name
            parameters = self._extract_parameters_from_tag(tag_name, model_info['parameter_count'])
            
            tags.append({
                'name': tag_name,
                'size': size_text,
                'size_bytes': size_bytes,
                'updated': updated_text,
                'quantization': quantization,
                'parameters': parameters
            })
        
        # Calculate estimated memory requirements
        for tag in tags:
            param_count = self._parse_parameter_count(tag['parameters'])
            quant_bits = self._get_quantization_bits(tag['quantization'])
            
            if param_count and quant_bits:
                # Base memory in GB (parameters * bits per parameter / 8 bits per byte / 1024^3)
                base_memory_gb = (param_count * quant_bits / 8) / (1024**3)
                
                # Add overhead for KV cache and other runtime requirements
                ram_required = round(base_memory_gb * 1.3, 1)  # 30% overhead
                vram_required = round(base_memory_gb * 1.2, 1)  # 20% overhead
                
                # Ensure minimum reasonable values
                ram_required = max(1.0, ram_required)
                vram_required = max(1.0, vram_required)
                
                tag['estimated_requirements'] = {
                    'ram_gb': ram_required,
                    'vram_gb': vram_required
                }
            else:
                tag['estimated_requirements'] = {
                    'ram_gb': None,
                    'vram_gb': None
                }
                
        # Update model_info with tags
        model_info['tags'] = tags
        
        return tags
    
    def _extract_parameters_from_tag(self, tag_name, default_parameters=None):
        """Extract parameter count from tag name if possible."""
        # Look for patterns like 7b, 13b, etc. in the tag name
        param_patterns = [
            r'[-:_](\d+[.,]?\d*)b',  # matches patterns like -7b, :13b, _70b, -1.1b
            r'(\d+[.,]?\d*)b\b'       # matches patterns like 7b, 13b at word boundary
        ]
        
        for pattern in param_patterns:
            match = re.search(pattern, tag_name.lower())
            if match:
                return f"{match.group(1)}B"
                
        return default_parameters or "Unknown"
    
    def _extract_readme(self, soup):
        """Extract README content from model page."""
        # Look for the README section
        readme_header = soup.find('h2', string='Readme')
        if not readme_header:
            return ""
            
        # Find the textarea containing the README
        textarea = readme_header.find_next('textarea')
        if textarea:
            return textarea.text.strip()
            
        return ""
    
    def _extract_parameter_count(self, soup, readme):
        """Extract parameter count from model page or README."""
        # First try to find it in the model specs on the page
        param_elem = soup.select_one('dt:-soup-contains("Parameters") + dd')
        if param_elem:
            return param_elem.text.strip()
            
        # Otherwise, try to find it in the README
        if readme:
            param_patterns = [
                r'(\d+(?:\.\d+)?)\s*[Bb]illion parameters',
                r'(\d+(?:\.\d+)?)\s*[Bb] parameters',
                r'(\d+(?:\.\d+)?)[Bb] parameter',
                r'parameters:\s*(\d+(?:\.\d+)?)\s*[Bb]'
            ]
            
            for pattern in param_patterns:
                match = re.search(pattern, readme)
                if match:
                    return f"{match.group(1)}B"
                    
        return None
    
    def _extract_context_window(self, soup, readme):
        """Extract context window size from model page or README."""
        # First try to find it in the model specs on the page
        context_elem = soup.select_one('dt:-soup-contains("Context") + dd')
        if context_elem:
            return context_elem.text.strip()
            
        # Otherwise, try to find it in the README
        if readme:
            context_patterns = [
                r'context(?:\s+window|\s+length|\s+size)?(?:\s+of)?\s+(\d[,\d]*)\s+tokens',
                r'(\d[,\d]*)\s+token\s+context',
                r'context\s+window:\s*(\d[,\d]*)'
            ]
            
            for pattern in context_patterns:
                match = re.search(pattern, readme, re.IGNORECASE)
                if match:
                    # Remove commas from the number
                    return match.group(1).replace(',', '')
                    
        return None
    
    def _parse_parameter_count(self, param_string):
        """Convert parameter string to actual number."""
        if not param_string or param_string == "Unknown":
            return None
            
        # Extract the number from strings like "7B", "13.5B"
        match = re.search(r'(\d+(?:\.\d+)?)', param_string)
        if match:
            return float(match.group(1)) * 1_000_000_000  # Convert billions to actual count
            
        return None
    
    def _get_quantization_bits(self, quant_string):
        """Convert quantization string to bits."""
        if not quant_string:
            return None
            
        # Extract the number from strings like "4-bit", "8-bit"
        match = re.search(r'(\d+)-bit', quant_string)
        if match:
            return int(match.group(1))
            
        return None
    
    def _fetch_page(self, url):
        """Fetch HTML content from a URL with error handling."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            sys.exit(1)
    
    def estimate_compatibility(self, system_specs, model_info):
        """
        Estimate if a model is compatible with the given system specs.
        
        Args:
            system_specs: Dictionary with system specifications
            model_info: Model information including tags
            
        Returns:
            Dictionary mapping tag names to compatibility results
        """
        results = {}
        
        # Extract system specs
        total_ram = system_specs.get('ram', 0)
        has_gpu = system_specs.get('has_gpu', False)
        gpu_vram = None
        if has_gpu and system_specs.get('gpu', {}).get('vram') != "Shared with system":
            gpu_vram = system_specs.get('gpu', {}).get('vram')
        
        # Check compatibility for each tag
        for tag in model_info.get('tags', []):
            tag_name = tag['name']
            reqs = tag.get('estimated_requirements', {})
            ram_required = reqs.get('ram_gb')
            vram_required = reqs.get('vram_gb')
            
            if not ram_required:
                results[tag_name] = {
                    'compatible': False,
                    'reason': "Unknown requirements"
                }
                continue
                
            # For GPU systems, check both RAM and VRAM
            if has_gpu and gpu_vram is not None:
                if total_ram >= ram_required and gpu_vram >= vram_required:
                    results[tag_name] = {
                        'compatible': True,
                        'reason': "Meets both RAM and VRAM requirements"
                    }
                elif gpu_vram < vram_required:
                    results[tag_name] = {
                        'compatible': False,
                        'reason': f"Not enough VRAM (need {vram_required}GB)"
                    }
                else:
                    results[tag_name] = {
                        'compatible': False,
                        'reason': f"Not enough RAM (need {ram_required}GB)"
                    }
            
            # For Apple Silicon (shared memory)
            elif has_gpu and system_specs.get('gpu', {}).get('vram') == "Shared with system":
                if total_ram >= max(ram_required, vram_required):
                    results[tag_name] = {
                        'compatible': True,
                        'reason': "Meets shared memory requirements"
                    }
                else:
                    results[tag_name] = {
                        'compatible': False,
                        'reason': f"Not enough shared memory (need {max(ram_required, vram_required)}GB)"
                    }
            
            # For CPU-only systems
            else:
                # CPU-only inference needs more RAM
                cpu_ram_required = ram_required * 1.2  # 20% more for CPU inference
                
                # Special case for small quantized models
                model_size_match = re.search(r'(\d+(?:\.\d+)?)', tag.get('parameters', '7B'))
                model_size = float(model_size_match.group(1)) if model_size_match else 7
                
                is_small_quantized = model_size <= 3 and tag.get('quantization', '') in ["2-bit", "3-bit", "4-bit"]
                is_medium_quantized = model_size <= 7 and tag.get('quantization', '') in ["2-bit", "3-bit", "4-bit"]
                
                if is_small_quantized and total_ram >= 4.0:
                    results[tag_name] = {
                        'compatible': True,
                        'reason': "Small quantized model runs with 4GB+ RAM"
                    }
                elif is_medium_quantized and total_ram >= 6.0:
                    results[tag_name] = {
                        'compatible': True,
                        'reason': "Medium quantized model runs with 6GB+ RAM"
                    }
                elif total_ram >= cpu_ram_required:
                    results[tag_name] = {
                        'compatible': True,
                        'reason': f"Meets CPU-only RAM requirement of {cpu_ram_required:.1f}GB"
                    }
                else:
                    results[tag_name] = {
                        'compatible': False,
                        'reason': f"Not enough RAM for CPU inference (need {cpu_ram_required:.1f}GB)"
                    }
        
        return results
        
    def save_to_json(self, filename="ollama_models.json"):
        """Save the collected model data to a JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.models, f, indent=2)
        print(f"Saved data to {filename}")

    def search_models(self, query):
        """
        Search for models using Ollama's search functionality
        
        Args:
                query: Search term (e.g., 'code', 'llama', etc.)
        """
        search_url = f"https://ollama.com/search?q={query}"
        print(f"Searching for models matching '{query}'...")
        
        html = self._fetch_page(search_url)
        
        # Parse the HTML to extract model cards
        soup = BeautifulSoup(html, 'html.parser')
        
        # First try the original selector
        model_cards = soup.select('a[href^="/library/"]')
        print(f"Found {len(model_cards)} model cards with first selector")
        
        # Try a more comprehensive selector that might catch additional models
        all_links = soup.find_all('a')
        library_links = [link for link in all_links if link.get('href') and link['href'].startswith('/library/')]
        print(f"Found {len(library_links)} model cards with broader selector")
        
        # Use whichever found more
        if len(library_links) > len(model_cards):
                model_cards = library_links
        
        # Let's check for duplicate models
        model_urls = set()
        unique_cards = []
        
        for card in model_cards:
                url = card['href']
                if url not in model_urls and not "/tags" in url:
                        model_urls.add(url)
                unique_cards.append(card)
        
        print(f"Found {len(unique_cards)} unique model cards")
        
        # Process each unique model card
        models = []
        for card in unique_cards:
                # Skip tags pages
                if "/tags" in card['href']:
                        continue
                
                # Get model name from URL
                model_name = card['href'].split('/')[-1]
                
                # Skip if this is not a model card
                if not model_name or len(model_name) < 2:
                        continue
                
                # Extract basic information from the card
                model_info = {
                'name': model_name,
                'url': f"{self.base_url}{card['href']}",
                'description': self._extract_description(card),
                'pull_count': self._extract_pull_count(card),
                'last_updated': self._extract_last_updated(card),
                'capabilities': self._extract_capabilities(card),
                'size': self._extract_size(card)
                }
                
                models.append(model_info)
        
        print(f"Found {len(models)} valid models matching '{query}'")
        return models


def get_system_specs():
    """Get basic system specifications"""
    # Get CPU info
    cpu_cores = psutil.cpu_count(logical=False)
    cpu_threads = psutil.cpu_count(logical=True)
    
    # Get CPU name
    cpu_name = "Unknown CPU"
    if platform.system() == "Darwin":  # macOS
        try:
            cpu_name = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        except:
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        cpu_name = line.split(":")[1].strip()
                        break
        except:
            pass
    elif platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            winreg.CloseKey(key)
        except:
            pass
    
    # Get RAM info
    total_ram = round(psutil.virtual_memory().total / (1024**3), 1)
    
    # Check for GPU
    has_gpu = False
    gpu_info = {"name": None, "vram": None}
    
    # Try NVIDIA GPU
    try:
        import subprocess
        nvidia_output = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]).decode()
        if nvidia_output.strip():
            has_gpu = True
            parts = nvidia_output.strip().split(',')
            gpu_info["name"] = parts[0].strip()
            gpu_info["vram"] = round(float(parts[1].strip()) / 1024, 1)  # Convert MiB to GiB
    except:
        pass
    
    # Check for Apple Silicon
    if not has_gpu and platform.system() == "Darwin" and platform.machine() == "arm64":
        has_gpu = True
        gpu_info["name"] = "Apple Silicon GPU"
        gpu_info["vram"] = "Shared with system"
    
    return {
        "cpu": {
            "name": cpu_name,
            "cores": cpu_cores,
            "threads": cpu_threads
        },
        "ram": total_ram,
        "has_gpu": has_gpu,
        "gpu": gpu_info,
        "os": platform.system()
    }

def main():
    """Main function."""
    import platform
    import subprocess
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape Ollama model library and check compatibility")
    parser.add_argument("--max-models", type=int, default=None, help="Maximum number of models to scrape (for testing)")
    parser.add_argument("--output", type=str, default="ollama_models.json", help="Output JSON file")
    parser.add_argument("--search", type=str, help="Search for models matching a specific term")
    args = parser.parse_args()
    
    # Get system specs
    system = get_system_specs()
    
    # Print system information
    print("System Information:")
    print(f"CPU: {system['cpu']['name']} ({system['cpu']['cores']} cores, {system['cpu']['threads']} threads)")
    print(f"RAM: {system['ram']}GB")
    
    if system["has_gpu"]:
        print(f"GPU: {system['gpu']['name']}")
        print(f"VRAM: {system['gpu']['vram']}GB" if system['gpu']['vram'] != "Shared with system" else f"VRAM: {system['gpu']['vram']}")
    else:
        print("GPU: None detected (CPU-only mode)")
    print()
    
    # Create the scraper
    scraper = OllamaLibraryScraper()
    
    # Scrape the main library page
    if args.search:
        models = scraper.search_models(args.search)
    else:
        models = scraper.scrape_library()
    
    # Limit the number of models to scrape if specified
    if args.max_models:
        models = models[:args.max_models]
    
    # Scrape detailed information for each model
    for i, model in enumerate(models):
        print(f"Scraping details for model {i+1}/{len(models)}: {model['name']}")
        scraper.scrape_model_details(model)
        
        # Check compatibility with the system
        compatibility = scraper.estimate_compatibility(system, model)
        model['compatibility'] = compatibility
        
        # Print compatible tags
        compatible_tags = [tag for tag, result in compatibility.items() if result['compatible']]
        if compatible_tags:
            print(f"  Compatible tags: {', '.join(compatible_tags)}")
        else:
            print("  No compatible tags found for this model")
            
    # Save all data to JSON
    scraper.save_to_json(args.output)
    
    # Count compatible models and tags
    compatible_models = {}
    for model in models:
        compatible_tags = [tag for tag, result in model.get('compatibility', {}).items() if result['compatible']]
        if compatible_tags:
            compatible_models[model['name']] = compatible_tags
    
    # Print summary
    print("\n=== COMPATIBILITY SUMMARY ===")
    print(f"Found {len(compatible_models)} compatible models out of {len(models)} total.")
    
    if compatible_models:
        print("\nCompatible Models:")
        for model_name, tags in compatible_models.items():
            print(f"• {model_name}")
            for tag in tags:
                # Find the tag details
                model_info = next((m for m in models if m['name'] == model_name), None)
                if model_info:
                    tag_info = next((t for t in model_info.get('tags', []) if t['name'] == tag), None)
                    if tag_info:
                        ram_required = tag_info.get('estimated_requirements', {}).get('ram_gb', 'Unknown')
                        print(f"  - {tag} (RAM: {ram_required}GB, {tag_info.get('quantization', 'Unknown')})")
                    else:
                        print(f"  - {tag}")
                else:
                    print(f"  - {tag}")
    else:
        print("\nNo compatible models found for your system.")




if __name__ == "__main__":
    main()