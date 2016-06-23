from lxml import html
from lxml import etree

import os
import sys
import re



def extract_text_node(n):
    # print(dir(n))
    t = ""
    for c in n.getchildren():
        if c.text is not None:
            t = t + c.text
    return t

def extract_votes(l):

    m = re.search("Don\'t Know: (.+?)%", l)
    if m:
        dont_know = m.group(1)
    else:
        dont_know = 0

    m = re.search("Mildly For: (.+?)%", l)
    if m:
        mildly_for = m.group(1)
    else:
        mildly_for = 0

    m = re.search("Mildly Against: (.+?)%", l)
    if m:
        mildly_against = m.group(1)
    else:
        mildly_against = 0

    m = re.search("Strongly Against: (.+?)%", l)
    if m:
        strongly_against = m.group(1)
    else:
        strongly_against = 0

    m = re.search("Strongly For: (.+?)%", l)
    if m:
        strongly_for = m.group(1)
    else:
        strongly_for = 0

    return (dont_know, mildly_for, mildly_against, strongly_against, strongly_for)

def build_xml(html_file, root):

    tree = html.fromstring(open(html_file,'r').read())
    links_results = tree.xpath('//*[@class="debate-stance-results"]/img/@src')
    flag_after = False
    for l in links_results:
        if re.search("After", l) is not None:
            flag_after = True

    if not(flag_after):
        return

    debateNode = etree.SubElement(root, "debate")
    title = tree.xpath('//*[@id="sidebar-side-1"]/div/div[1]/div[1]/h1/text()')

    titleNode = etree.SubElement(debateNode, "title")
    if title != []:
        titleNode.text = title[0]

    abstract = tree.xpath('//*[@id="sidebar-side-1"]/div/div[1]/div[1]/div[1]/div/p/text()')
    abstractNode = etree.SubElement(debateNode, "abstract")
    if abstract != []:
        abstractNode.text = abstract[0]

    points = tree.xpath('//div[@class="debate_points_arguments"]')

    all_points_for = []

    all_points_against = []

    print(html_file)
    if points != []:
        # looking for points and counterpoints in arguments for
        points_for = points[0].xpath('//*[@id="group1"]')
        titles_points_for = points_for[0].xpath('//*[@class="debate_for_point  debate_point"]/h4/text()')
        arg_for = points_for[0].xpath('//*[@class="debate_for_argument_point debate_argument_item "]/div[@class="pointinner"]')
        arg_for_txt = []
        for a in arg_for:
            arg_for_txt.append(extract_text_node(a))

        counter_for = points_for[0].xpath('//*[@class="debate_for_argument_counter debate_argument_item"]/div[@class="pointinner"]')
        counter_for_txt = []
        for a in counter_for:
            counter_for_txt.append(extract_text_node(a))

        # setting arguments for, points and counter points
        all_points_for = list(zip(titles_points_for, arg_for_txt, counter_for_txt))
        
        # looking for points and counterpoints in arguments against
        points_against = points[0].xpath('//*[@id="group2"]')
        titles_points_against = points_against[0].xpath('//*[@class="debate_against_point  debate_point"]/h4/text()')
        arg_against = points_against[0].xpath('//*[@class="debate_against_argument_point debate_argument_item "]/div[@class="pointinner"]')
        arg_against_txt = []
        for a in arg_against:
            arg_against_txt.append(extract_text_node(a))

        counter_against = points_against[0].xpath('//*[@class="debate_against_argument_counter debate_argument_item"]/div[@class="pointinner"]')
        counter_against_txt = []
        for a in counter_against:
            counter_against_txt.append(extract_text_node(a))

        # setting arguments against, points and counter points
        all_points_against = list(zip(titles_points_against, arg_against_txt, counter_against_txt))

    argNode = etree.SubElement(debateNode, "arguments")
    proNode = etree.SubElement(argNode, "pro")
    againstNode = etree.SubElement(argNode, "against")

    for (t,p,c) in all_points_for:
        itemNode = etree.SubElement(proNode, "item")
        descNode = etree.SubElement(itemNode, "desc")
        descNode.text = t
        pointNode = etree.SubElement(itemNode, "point")
        pointNode.text = p
        counterNode = etree.SubElement(itemNode, "counterpoint")
        counterNode.text = c

    for (t,p,c) in all_points_against:
        itemNode = etree.SubElement(againstNode, "item")
        descNode = etree.SubElement(itemNode, "desc")
        descNode.text = t
        pointNode = etree.SubElement(itemNode, "point")
        pointNode.text = p
        counterNode = etree.SubElement(itemNode, "counterpoint")
        counterNode.text = c

    links_results = tree.xpath('//*[@class="debate-stance-results"]/img/@src')
    flag_before = False
    flag_after = False
    for l in links_results:
        if re.search("Before", l) is not None:
            (bdont_know, bmildly_for, bmildly_against, bstrongly_against, bstrongly_for) = extract_votes(l)
            flag_before = True

        if re.search("After", l) is not None:
            (adont_know, amildly_for, amildly_against, astrongly_against, astrongly_for) = extract_votes(l)
            flag_after = True

    votesNode = etree.SubElement(debateNode, "votes")
    beforeNode = etree.SubElement(votesNode, "before")

    if flag_before:
        bdont_knowNode = etree.SubElement(beforeNode, "dontknow")
        bdont_knowNode.text = str(bdont_know)
        bmildly_forNode = etree.SubElement(beforeNode, "mildlyfor")
        bmildly_forNode.text = str(bmildly_for)
        bmildly_againstNode = etree.SubElement(beforeNode, "mildlyagainst")
        bmildly_againstNode.text = str(bmildly_against)
        bstrongly_forNode = etree.SubElement(beforeNode, "stronglyfor")
        bstrongly_forNode.text = str(bstrongly_for)
        bstrongly_againstNode = etree.SubElement(beforeNode, "stronglyagainst")
        bstrongly_againstNode.text = str(bstrongly_against)

    afterNode = etree.SubElement(votesNode, "after")
    if flag_after:
        adont_knowNode = etree.SubElement(afterNode, "dontknow")
        adont_knowNode.text = str(adont_know)
        amildly_forNode = etree.SubElement(afterNode, "mildlyfor")
        amildly_forNode.text = str(amildly_for)
        amildly_againstNode = etree.SubElement(afterNode, "mildlyagainst")
        amildly_againstNode.text = str(amildly_against)
        astrongly_forNode = etree.SubElement(afterNode, "stronglyfor")
        astrongly_forNode.text = str(astrongly_for)
        astrongly_againstNode = etree.SubElement(afterNode, "stronglyagainst")
        astrongly_againstNode.text = str(astrongly_against)
       

def process_data(dir_data, output):

    root = etree.Element("debates")
    for subdir, dirs, files in os.walk(dir_data):
        for f in files:
            if f.endswith(".html"):
                # debateNode = etree.SubElement(root, "debate")
                html_fname = os.path.join(subdir, f)
                build_xml(html_fname, root)

    fd = open(output, "w")
    fd.write(etree.tostring(root, pretty_print=True).decode('utf-8'))
    fd.close()
        
if __name__ == "__main__":
    dir_data = sys.argv[1]
    output = "idebate_top100Votes.xml"
    process_data(dir_data, output)
