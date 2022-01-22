# Wildfire
This directory contains scripts and notebooks that help with analyzing how twitter can provide a source of information to help analyze the health impacts of wildfires. Please adhere to the following principles when using this repository:
* Add notebooks under the notebooks subfolder.
* If you find yourself rewriting the same code more than once, turn it into a script and put it under the "scripts" directory.
* If you save any data, please add it to a separate directory called "data." This file is already added to the .gitignore folder, so it will not be pushed to the repository. This means that data will not be backed up.

## What this file does not contain
Please reference the [.gitignore](.gitignore) file to see what will be ommitted when you push to this directory. Any data files should be ignored to minimize the directory size. The [.gitignore](.gitignore) file is configured to recognize these files and hopefully ignore them, but please make sure that you check the files being tracked before you commit to the repository. 


## Referencing scripts from notebooks
Since we have our scripts saved in a different folder, we have to add the script to our path so we can leverage that code:

```python
# This will import the twitter code into your path.
import sys
sys.path.append("../scripts/twitter")
from searchTwitter import TwitterDataFrame, TwitterSearchTerm
```