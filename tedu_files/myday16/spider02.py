import re

html = '''
<div class="wrap">
                        <ul>
                            <li>螺纹钢</li>
                            <li>HRB400E</li>
                            <li class="">3990</li>
                        </ul>
                        <ul>
                            <li>Φ14mm</li>
                            <li>永钢</li>
                            <li class="">-</li>
                        </ul>
                    </div>
'''
re_pp = '<div class="wrap">.*?<li>(.*?)</li>.*?<li>(.*?)</li>.*?<li class="">(.*?)</li>.*?<li>(.*?)</li>.*?<li>(.*?)</li>'
p = re.compile(re_pp,re.S)
r_list = p.findall(html)
print(r_list)
