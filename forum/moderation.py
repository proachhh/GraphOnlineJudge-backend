BANNED_KEYWORDS = [
    # ===== 色情内容 =====
    '成人网站', '成人电影', '黄色网站', '色情', 'av女优',
    '三级片', '裸聊', '约炮', '一夜情', '福利姬',
    '嫖娼', '卖淫', '招嫖', '外围女',
    '色情直播', '色情视频', '成人直播',
    '性交', '口交', '肛交', '轮奸', '强奸',
    '无码', '有码', '中出', '潮吹', '颜射',

    # ===== 赌博相关 =====
    '赌博', '赌场', '博彩', '六合彩', '时时彩',
    '棋牌平台', '现金游戏', '网赌',
    '百家乐', '真人视讯', '赌球',
    'ag真人', 'bbin', 'mg电子', 'pg电子', 'pp电子',
    '新葡京', '澳门银河', '威尼斯人',
    '赌博网站', '赌博平台', '赌博app', '线上赌场',

    # ===== 毒品相关 =====
    '毒品', '大麻', '冰毒', '海洛因', '可卡因',
    '摇头丸', 'k粉', '麻古', '杜冷丁',
    '吸毒', '贩毒', '制毒', '溜冰',
    '毒贩', '毒品交易', '毒品货源',

    # ===== 暴力恐怖 =====
    '恐怖分子', 'isis', '伊斯兰国', '基地组织',
    '杀人', '买凶', '雇凶',
    '持枪', '枪支', '手枪', '步枪', '枪械',
    '炸弹', '爆炸', '炸药', '雷管',

    # ===== 诈骗广告 =====
    '兼职日结', '刷单', '刷信誉', '代开发票',
    '代理充值', '充值卡', '话费卡',
    '加微信', '微信号', '加qq', 'qq群',
    '联系客服', '点击领取', '免费领取',
    '网贷', '贷款', '信用卡套现',
    '人肉搜索', '人肉',

    # ===== 辱骂攻击 =====
    '操你妈', '草泥马', 'fuck', 'fuck you',
    'nmsl', 'cnm', 'ntmd', 'sb',
    '傻逼', '傻b', '弱智', '脑残',
    '妈逼', '你妈', '他妈', '我操',
    '日你', '狗日的', '龟儿子', '婊子',
    '杂种', '畜牲', '畜生', '王八蛋',
    '妈的', '妈的逼', '他妈的',
    '肏', '屄', '屌', '靠',
    'bitch', 'dick', 'asshole', 'bastard',

    # ===== 政治敏感 =====
    '法轮功', '法轮大法', 'flg', 'zuzhi',
    '六四', '天安门事件','六四事件','习近平','胡静涛',
    'xjp', '习明泽', '平西王',
]


def _strip_html(text):
    """去除HTML标签，提取纯文本"""
    import re
    return re.sub(r'<[^>]+>', '', text)


def contains_banned(text):
    if not text:
        return False, None
    plain = _strip_html(text).lower()
    for word in BANNED_KEYWORDS:
        if word.lower() in plain:
            return True, word
    return False, None


def moderate_content(title, content, user=None):
    """审核标题和内容，返回 (通过, 违规词)。管理员跳过检测。"""
    if user and user.is_authenticated and user.is_admin_role():
        return True, None
    for field in [title, content]:
        found, word = contains_banned(field)
        if found:
            return False, word
    return True, None
