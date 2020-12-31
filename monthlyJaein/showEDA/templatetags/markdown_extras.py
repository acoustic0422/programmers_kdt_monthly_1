from django import template
from django.template.defaultfilters import stringfilter

import markdown as md

from markdownx.utils import markdownify

register = template.Library()


@register.filter()
@stringfilter
def markdown(value):
    return md.markdown(value, extensions=['markdown.extensions.fenced_code'])


@register.filter()
@stringfilter
def convert_markdown(value):
    return markdownify(value)