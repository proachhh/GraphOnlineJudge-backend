import re
import io
import os
import json
import zipfile
import logging
import requests
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

LOJ_API_ENDPOINT = 'https://api.loj.ac/'
LOJ_API_TIMEOUT = 30


def fetch_loj_problem(problem_id: int) -> dict:
    """直接调用 LOJ API 获取题目完整数据"""
    url = f'{LOJ_API_ENDPOINT}api/problem/getProblem'
    body = {
        'displayId': problem_id,
        'localizedContentsOfLocale': 'zh_CN',
        'samples': True,
        'judgeInfo': True,
        'tagsOfLocale': 'zh_CN',
    }
    resp = requests.post(url, json=body, timeout=LOJ_API_TIMEOUT,
                         headers={'Content-Type': 'application/json'})
    resp.raise_for_status()
    return resp.json()


def fetch_loj_testcases(problem_id: int) -> str:
    """下载 LOJ 题目的测试点文件，返回 zip 文件路径。

    LOJ 测试点的输出文件后缀可能是 .out / .ans / .res（不同题目配置不同）。
    本 OJ 仅识别 .out / .res，因此将 .ans 归一化为 .out 后再打包；
    并按 base_name 将每个 .in 与其对应输出配对，跳过无法配对的孤立文件。
    """
    url = f'{LOJ_API_ENDPOINT}api/problem/downloadProblemFiles'
    body = {
        'problemId': problem_id,
        'type': 'TestData',
        'filenameList': [],
    }
    resp = requests.post(url, json=body, timeout=LOJ_API_TIMEOUT,
                         headers={'Content-Type': 'application/json'})
    resp.raise_for_status()
    data = resp.json()

    download_info = data.get('downloadInfo', [])
    if not download_info:
        return None

    # 收集 .in 文件与输出文件（.out/.ans/.res），按 base_name 索引
    in_urls = {}        # base_name -> download_url
    out_urls = {}       # base_name -> (ext, download_url)
    for item in download_info:
        filename = item.get('filename', '')
        dl_url = item.get('downloadUrl', '')
        if not filename or not dl_url:
            continue
        if filename.endswith('.in'):
            in_urls[filename[:-3]] = dl_url
        elif filename.endswith('.out'):
            out_urls.setdefault(filename[:-4], ('.out', dl_url))
        elif filename.endswith('.ans'):
            # .ans 与 .out 同义，归一化为 .out
            out_urls.setdefault(filename[:-4], ('.out', dl_url))
        elif filename.endswith('.res'):
            out_urls.setdefault(filename[:-4], ('.res', dl_url))

    # 按 base_name 配对：.in + 对应输出；.ans 在写入时改名为 .out
    pairs = []  # list of (filename_in_zip, download_url)
    for base in sorted(in_urls.keys()):
        in_url = in_urls[base]
        out = out_urls.get(base)
        if not out:
            # 无对应输出（如 SPJ/交互题仅输入），跳过——此类题目需手动处理
            continue
        ext, out_url = out
        pairs.append((base + '.in', in_url))
        pairs.append((base + ext, out_url))

    if not pairs:
        return None

    # 并发下载到临时目录（流式写盘，避免大测试点撑爆内存），再打包成 zip。
    # .ans 的内容写到 .out 名下。
    import tempfile
    import shutil
    tmp_dir = tempfile.mkdtemp(prefix='loj_tc_')

    def _fetch(pair):
        filename, dl_url = pair
        # 防御路径穿越：LOJ 文件名形如 "1.in"，仅取 basename
        safe = os.path.basename(filename)
        if not safe or safe != filename or '/' in filename or '\\' in filename:
            return filename, False
        dst = os.path.join(tmp_dir, safe)
        try:
            with requests.get(dl_url, timeout=120, stream=True) as r:
                r.raise_for_status()
                with open(dst, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
            return filename, True
        except Exception as e:
            logger.warning(f'Failed to download {filename}: {e}')
            return filename, False

    ok_files = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        for filename, ok in ex.map(_fetch, pairs):
            if ok:
                ok_files.append(filename)

    tmp_path = tempfile.mktemp(suffix='.zip')
    try:
        with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for filename in ok_files:
                src = os.path.join(tmp_dir, filename)
                if os.path.isfile(src):
                    zf.write(src, filename)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return tmp_path


def parse_loj_url(url: str) -> dict:
    s = (url or '').strip()

    # 允许直接输入数字编号，例如 "5" 等价于 "https://loj.ac/p/5"
    if s.isdigit():
        problem_id = int(s)
    else:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if 'loj.ac' not in host and 'libreoj' not in host:
            raise ValueError(f'不支持的 OJ 网站: {host or "空"}，请输入 LOJ 题目编号或 loj.ac 链接')
        path_match = re.match(r'/p/(\d+)', parsed.path)
        if not path_match:
            raise ValueError('无法提取题目 ID，请输入 LOJ 题目编号（如 5）或 https://loj.ac/p/数字')
        problem_id = int(path_match.group(1))

    return {
        'source': 'loj',
        'problem_id': problem_id,
        'api_url': f'{LOJ_API_ENDPOINT}api/problem/getProblem',
        'api_method': 'POST',
        'api_body': {
            'displayId': problem_id,
            'localizedContentsOfLocale': 'zh_CN',
            'samples': True,
            'judgeInfo': True,
            'tagsOfLocale': 'zh_CN',
        },
    }


def _esc_html(text):
    """转义 HTML 特殊字符"""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _nl_to_br(text):
    """将换行符 \\n 转换为 <br>，保留前端 KaTeX 数学公式不受影响。

    LOJ 返回的题目内容是 markdown 纯文本（以 \\n 换行），而前端使用 v-html
    渲染时会忽略 \\n 换行符。此函数仅将换行转换为 <br>，不改变其他内容，
    保持与原有渲染行为一致（$ 公式仍由前端 KaTeX 处理）。
    """
    if not text or not text.strip():
        return text or ''
    # 如果内容已经包含 HTML 块级标签，视为已转换，直接返回
    if re.search(r'<(?:p|div|br\s*/?|h[1-6]|ul|ol|li|table|pre|blockquote)\b', text, re.I):
        return text

    # 保护数学公式（$$...$$, $...$, \[...\], \(...\)）不被 <br> 破坏
    placeholders = []

    def _save(m):
        idx = len(placeholders)
        placeholders.append(m.group(0))
        return f'\x00PH{idx}\x00'

    tmp = re.sub(r'\$\$([\s\S]*?)\$\$', _save, text)
    tmp = re.sub(r'(?<!\$)\$(?!\$)([^\$\n]+?)\$', _save, tmp)
    tmp = re.sub(r'\\\[([\s\S]*?)\\\]', _save, tmp)
    tmp = re.sub(r'\\\(([\s\S]*?)\\\)', _save, tmp)

    # 换行符 → <br>
    tmp = tmp.replace('\n', '<br>')

    # 恢复数学公式
    tmp = re.sub(r'\x00PH(\d+)\x00', lambda m: placeholders[int(m.group(1))], tmp)
    return tmp


def parse_loj_api_json(raw) -> dict:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError('JSON 格式无效，请确认粘贴的是 LOJ API 返回的完整 JSON')

    meta = raw.get('meta', raw)
    localized = raw.get('localizedContentsOfLocale', {})
    title = str(localized.get('title', ''))
    if not title:
        title = str(meta.get('title', ''))
    if isinstance(title, dict):
        title = title.get('zh_CN', str(title))

    # 按 section title 分类内容
    description = ''
    input_desc = ''
    output_desc = ''
    hint = ''
    sections = localized.get('contentSections', [])
    for sec in sections:
        sec_title = sec.get('sectionTitle', '')
        sec_text = sec.get('text', sec.get('html', ''))
        sec_type = sec.get('type', '')
        t = sec_title.lower() if sec_title else ''
        if '题目描述' in t or '描述' in t or '背景' in t:
            description = sec_text
        elif '输入' in t:
            input_desc = sec_text
        elif '输出' in t:
            output_desc = sec_text
        elif '样例' in t:
            # 样例说明文本，附加到 hint
            if sec_text:
                hint = (hint + '\n\n' if hint else '') + f'**样例说明**\n{sec_text}'
        elif '提示' in t or '数据范围' in t or '范围' in t or '说明' in t:
            hint = (hint + '\n\n' if hint else '') + sec_text
        elif sec_type == 'Text':
            # 其他文本段，附加到 description
            description = (description + '\n\n' if description else '') + sec_text

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

    problem_id = str(meta.get('id', raw.get('id', '')))

    return {
        'title': title,
        'description': _nl_to_br(description),
        'input_description': _nl_to_br(input_desc),
        'output_description': _nl_to_br(output_desc),
        'samples': samples,
        'hint': _nl_to_br(hint),
        'time_limit': time_limit,
        'memory_limit': memory_limit,
        'difficulty': difficulty,
        'tags': tags,
        'source': f'LOJ #{problem_id}' if problem_id else 'LOJ',
        'source_oj': 'loj',
        'problem_id': problem_id,
    }
