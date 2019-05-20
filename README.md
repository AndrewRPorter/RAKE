RAKE
====

This is a fork of [RAKE](https://github.com/aneesha/RAKE) with a focus on speed and readability.

This fork uses a different `StopList` and utilizes a large english word corpus to generate more accurate and useful phrase scores. The corpus can be purchased at [Word frequency data](https://www.wordfrequency.info/free.asp).

Usage
=====

```python
r = Rake()
print(r.get_phrases("Some input text..."))
print(r.get_phrases("Some input text...", length=5))  # explicitly define phrase list length
```