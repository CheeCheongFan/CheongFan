def htmlcleanerBS(text):
    blacklist = [
        'style',
        'script',
        # other elements,
    ]
    text_elements = [t for t in text if t.parent.name not in blacklist]
    whitelist = [
        'p',

    ]
    text_elements2 = [t for t in text_elements if t.parent.name in whitelist]
    return text_elements2