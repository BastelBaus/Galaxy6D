# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Galaxy6D'
copyright = '2024, BastelBaus'
author = 'BastelBaus'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [ 'sphinx.ext.autodoc', 'recommonmark']




templates_path = ['_templates']
exclude_patterns = ['archive','sources','photos']

import sys, os
#sys.path.insert(0, os.path.abspath('../Galaxy6DLib/src'))


#source_suffix = ['.rst', '.md']
source_suffix = ['.rst']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = "pydata_sphinx_theme"

html_static_path = ['_static']

#html_theme = 'classic'
#html_theme_options = {
#    "stickysidebar": "true", 
#    }
    
#def setup(app):
#    app.add_css_file('my_theme.css')    


html_theme_options = {
    "header_links_before_dropdown": 2,
    "icon_links": [
        {
            "name": "hackaday",
            "url": "https://hackaday.io/project/192855",
            "icon": "https://hackaday.io/favicon.ico",
            "type": "url",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/BastelBaus/Galaxy6D",
            "icon": "fa-brands fa-github",
        },

    ]
}


