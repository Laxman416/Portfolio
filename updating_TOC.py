"""
A python file to automatically search for files and update the README.md file when the script is run.

"""

import os

def generate_toc():
    base_path = 'Python'  # Base directory containing your notes and challenges
    toc_lines = ["## Table of Contents\n"]  # Initialize the TOC lines

    # Python Notes
    toc_lines.append("### Python\n")
    toc_lines.append("A collection of notes and challenges for Python programming.\n")
    
    # Add Python Notes section
    toc_lines.append("**Python Notes**  \n")
    notes_path = os.path.join(base_path, 'Notes')
    if os.path.exists(notes_path):
        for filename in sorted(os.listdir(notes_path)):
            if filename.endswith('.md'):  # Only include markdown files
                file_path = os.path.join(notes_path, filename)
                # Add to TOC
                toc_lines.append(f"- [{filename[:-3]}]({file_path})  \n")

    # Add Python Challenges section
    toc_lines.append("\n**Python Challenges**  \n")
    challenges_path = os.path.join(base_path, '8kyu')  # Adjust as needed for other kyu levels
    if os.path.exists(challenges_path):
        toc_lines.append("- **8 kyu**  \n")
        for filename in sorted(os.listdir(challenges_path)):
            if filename.endswith('.py'):  # Only include Python files
                file_path = os.path.join(challenges_path, filename)
                # Add to TOC
                toc_lines.append(f"    - [{filename[:-3]}]({file_path})  \n")
    
    # Write the updated TOC back to README.md
    readme_path = 'README.md'
    with open(readme_path, 'r+') as readme_file:
        content = readme_file.readlines()
        
        # Find the index to insert TOC
        toc_start_index = -1
        toc_end_index = -1
        
        for i, line in enumerate(content):
            if line.startswith("## Table of Contents"):
                toc_start_index = i
            if toc_start_index != -1 and line.strip() == "":
                toc_end_index = i
                break
        
        if toc_start_index != -1:
            # Update the TOC in README.md
            content = content[:toc_start_index + 1] + toc_lines + content[toc_end_index:]
            readme_file.seek(0)
            readme_file.writelines(content)
            readme_file.truncate()  # Remove any leftover content after the new TOC

    print("Table of Contents updated successfully!")

if __name__ == "__main__":
    generate_toc()
