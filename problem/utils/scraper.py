import re
import json
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

LOJ_API_ENDPOINT = 'https://api.loj.ac/'


def parse_loj_url(url: str) -> dict:
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    if 'loj.ac' not in host and 'libreoj' not in host:
        raise ValueError(f'不支持的 OJ 网站: {host}，目前仅支持 loj.ac')

    path_match = re.match(r'/p/(\d+)', parsed.path)
    if not path_match:
        raise ValueError('无法从 URL 中提取题目 ID，URL 格式应为 https://loj.ac/p/数字')

    problem_id = int(path_match.group(1))

    return {
        'source': 'loj',
        'problem_id': problem_id,
        'api_url': f'{LOJ_API_ENDPOINT}d/loj/problem/getProblem',
        'api_method': 'POST',
        'api_body': {
            'displayId': problem_id,
            'localizedContentsOfLocale': 'zh_CN',
            'samples': True,
            'judgeInfo': True,
            'tagsOfLocale': 'zh_CN',
        },
    }


def parse_loj_api_json(raw) -> dict:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError('JSON 格式无效，请确认粘贴的是 LOJ API 返回的完整 JSON')

    meta = raw.get('meta', raw)

    localized = raw.get('localizedContentsOfLocale', {})
    title = str(localized.get('title', ''))
    description = ''
    if localized:
        sections = localized.get('contentSections', [])
        if sections:
            parts = []
            for sec in sections:
                sec_title = sec.get('sectionTitle', '')
                sec_text = sec.get('text', sec.get('html', ''))
                if sec_title:
                    parts.append(f'<h3>{sec_title}</h3>')
                parts.append(sec_text)
            description = '\n'.join(parts)

    if not title:
        title = str(meta.get('title', raw.get('title', '')))
    if isinstance(title, dict):
        title = title.get('zh_CN', str(title))
    if not description:
        desc_raw = meta.get('description', raw.get('data', {}).get('description', ''))
        if isinstance(desc_raw, dict):
            desc_raw = desc_raw.get('zh_CN', str(desc_raw))
        description = str(desc_raw) if desc_raw else ''

    input_desc = ''
    output_desc = ''
    id_raw = meta.get('inputDescription', raw.get('data', {}).get('inputDescription', raw.get('inputDescription', '')))
    input_desc = str(id_raw) if id_raw and not isinstance(id_raw, dict) else ''
    od_raw = meta.get('outputDescription', raw.get('data', {}).get('outputDescription', raw.get('outputDescription', '')))
    output_desc = str(od_raw) if od_raw and not isinstance(od_raw, dict) else ''

    hint_raw = raw.get('hint', meta.get('hint', ''))
    hint = str(hint_raw) if hint_raw and not isinstance(hint_raw, dict) else ''

    def _int(val, default=0):
        try: return int(val)
        except (TypeError, ValueError): return default

    judge = raw.get('judgeInfo', {})
    time_limit = _int(judge.get('timeLimit', meta.get('timeLimit', 1000)), 1000)
    memory_limit = _int(judge.get('memoryLimit', meta.get('memoryLimit', 256)), 256)

    diff_map = {'Low': 'Low', 'Mid': 'Mid', 'High': 'High', 1: 'Low', 2: 'Mid', 3: 'High'}
    raw_diff = meta.get('difficulty', raw.get('difficulty', 'Mid'))
    difficulty = diff_map.get(raw_diff, 'Mid')

    tags = []
    tags_of_locale = raw.get('tagsOfLocale', raw.get('tags', meta.get('tags', [])))
    if isinstance(tags_of_locale, list):
        for t in tags_of_locale:
            if isinstance(t, str):
                tags.append(t)
            elif isinstance(t, dict):
                tags.append(str(t.get('name', t.get('localizedName', str(t)))))

    samples = []
    samples_raw = raw.get('samples', meta.get('samples', [])) or []
    for s in samples_raw:
        inp = str(s.get('input', s.get('inputData', '')))
        out = str(s.get('output', s.get('outputData', '')))
        samples.append({'input': inp, 'output': out})

    source_url = str(meta.get('url', raw.get('url', '')))
    problem_id = str(meta.get('id', raw.get('id', '')))

    return {
        'title': title,
        'description': description,
        'input_description': input_desc,
        'output_description': output_desc,
        'samples': samples,
        'hint': hint,
        'time_limit': time_limit,
        'memory_limit': memory_limit,
        'difficulty': difficulty,
        'tags': tags,
        'source': source_url,
        'source_oj': 'loj',
        'problem_id': problem_id,
    }
