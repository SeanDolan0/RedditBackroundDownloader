# Reddit Background Downloader

A lightweight Python CLI tool that downloads high-resolution Reddit images, intelligently crops them for wallpaper use (16:9 aspect ratio), and saves them with Windows-safe filenames. Features an interactive mode for beginners and traditional CLI for automation.

## ✨ Features

- **Interactive Mode** - Answer simple prompts instead of remembering CLI flags (`python scraper.py -i`)
- **Beautiful CLI Output** - Colored tables, progress notifications, and professional formatting
- **No API Credentials Required** - Uses Reddit's public JSON endpoint
- **Smart Content-Aware Cropping** - Saliency detection + optional face detection
- **Center Crop Mode** - Simple fallback for straightforward cropping
- **Resolution Guardrails** - Enforces minimum 2560×1440 for wallpaper quality
- **Clean Processing** - Minimal, quiet output with detailed final summary
- **Windows-Safe Filenames** - Automatically sanitizes filenames and prevents collisions
- **Cross-Platform** - Works on Windows, macOS, and Linux

## 🚀 Quick Start

### For Beginners (Interactive Mode)
```bash
python scraper.py -i
```
Simply answer the prompts and let it run!

### For Scripts/Automation (CLI Mode)
```bash
python scraper.py --subreddit EarthPorn --limit 20 --sort hot
```

### Get Help
```bash
python scraper.py --help
```

## 📋 Requirements

- Python 3.10 or higher
- Dependencies listed in `requirements.txt`

## 📦 Installation

### First Time Setup
```bash
# Clone or download the repository
cd Reddit-downloader

# Install dependencies
pip install -r requirements.txt
```

### Updating Dependencies
If you already have the repo installed, update with:
```bash
pip install -r requirements.txt --upgrade
```

## 📖 Usage

### Interactive Mode (Recommended for beginners)

Launch the interactive mode:

```bash
python scraper.py --interactive
```

or shorthand:

```bash
python scraper.py -i
```

You'll be prompted for:
1. **Subreddit name** - Which subreddit to scrape (e.g., `EarthPorn`)
2. **Image count** - How many images to download
3. **Sort method** - `hot`, `new`, or `top`
4. **Time range** - If sorting by `top` (hour, day, week, month, year, all)
5. **Crop mode** - `smart` (AI-based) or `center` (simple center crop)
6. **Output folder** - Where to save the images

### Command Line Mode

#### Basic Usage
```bash
python scraper.py --subreddit EarthPorn --limit 20
```

#### Top Posts All Time
```bash
python scraper.py --subreddit EarthPorn --limit 20 --sort top --time all
```

#### Center Crop Mode
```bash
python scraper.py --subreddit EarthPorn --limit 20 --sort hot --crop-mode center
```

#### Custom Output Folder
```bash
python scraper.py --subreddit EarthPorn --limit 20 --output ./my_wallpapers
```

#### Full Example
```bash
python scraper.py \
  --subreddit EarthPorn \
  --limit 50 \
  --sort top \
  --time all \
  --crop-mode smart \
  --output "C:\Users\[User]\Pictures\Wallpapers"
```

## 🎯 CLI Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--interactive` | `-i` | N/A | Launch interactive mode with guided prompts |
| `--subreddit` | N/A | Required (CLI) | Subreddit name to scrape (without `r/`) |
| `--limit` | N/A | `50` | Number of images to save |
| `--sort` | N/A | `hot` | Sort method: `hot`, `new`, or `top` |
| `--time` | N/A | `all` | Time range for `--sort top`: `hour`, `day`, `week`, `month`, `year`, `all` |
| `--crop-mode` | N/A | `smart` | Cropping strategy: `smart` or `center` |
| `--output` | N/A | `./output_images` | Destination folder for saved images |
| `--help` | `-h` | N/A | Show help message |

## 📊 Output

The tool provides clean, organized output:

1. **Welcome Banner** - Styled title
2. **Configuration Table** - Your selected settings
3. **Processing Messages** - Status updates and save confirmations
4. **Summary Table** - Detailed statistics including:
   - Elapsed time
   - Pages fetched from Reddit
   - Posts examined
   - Image candidates found
   - Download failures
   - Successfully saved images
   - Images skipped (too small, no image, errors)

## 💡 Tips

- **`--limit` refers to saved images**, not posts fetched. The script will page through Reddit until it reaches your target or runs out of usable posts.
- **Face detection** - If unavailable in your environment, the script automatically falls back to saliency-based cropping.
- **Output folder** - Created automatically if it doesn't exist.
- **Filenames** - Include post ID to reduce collisions, e.g., `Beautiful_Mountain_1a2b3c_2560x1440.jpg`
- **Smart Crop** - Uses saliency maps to identify interesting regions, optionally boosted by face detection
- **Center Crop** - Simple centered crop, good for landscapes and predictable compositions

## 🔧 How It Works

1. **Fetch Posts** - Retrieves posts from the specified subreddit using Reddit's public JSON API
2. **Find Images** - Extracts image URLs from common Reddit image hosts
3. **Download** - Asynchronously downloads high-resolution image files
4. **Crop** - Intelligently crops to 16:9 aspect ratio (2560×1440 minimum)
5. **Save** - Stores images with sanitized, collision-free filenames
6. **Report** - Displays comprehensive summary statistics

## 📂 File Structure

```
Reddit-downloader/
├── scraper.py              # Main script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── LICENSE                # License
└──.gitignore             # Git exclusions
```

## 🤝 Contributing

Found a bug or have a feature idea? Feel free to open an issue or submit a pull request!

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚖️ Legal Notice

- This tool uses Reddit's **public** JSON endpoint - no credentials needed
- Only downloads publicly available content
- Respects Reddit's terms of service by using appropriate user-agents
- For bulk downloads, consider reaching out to Reddit for guidance

## 🐛 Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### Face detection not working
- This is normal! The script automatically falls back to saliency-based cropping
- No functionality is lost

### Getting fewer images than requested
- The subreddit might not have enough high-resolution images meeting the criteria
- Try a different subreddit or increase the `--limit` value

### Output folder permission denied
- Ensure you have write permissions to the specified output folder
- Try using `./output_images` instead of a full path

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 2 minutes
- **[CLEANUP.md](CLEANUP.md)** - Learn about the clean output design
- **[DEMO.md](DEMO.md)** - See visual examples of the output

## 🎉 Enjoy!

Download beautiful wallpapers with ease!
