import os
import statsd

from notion_client import Client


notion_token = os.getenv("NOTION_TOKEN")
notion = Client(auth=notion_token)

s = statsd.StatsClient("localhost", 8125)

d = {
    "d216d939579c4406872066f76f4aebe6": "ai_reading_list",
    "f3515674357743debbb199bec6da454b": "system_reading_list",
}


def get_block_content(block):
    block_type = block["type"]
    if block_type == "paragraph":
        return (
            "1+" + block["paragraph"]["rich_text"][0]["plain_text"]
            if block["paragraph"]["rich_text"]
            else ""
        )
    elif block_type == "heading_1":
        return (
            "2+" + block["heading_1"]["rich_text"][0]["plain_text"]
            if block["heading_1"]["rich_text"]
            else ""
        )
    elif block_type == "heading_2":
        return (
            "3+" + block["heading_2"]["rich_text"][0]["plain_text"]
            if block["heading_2"]["rich_text"]
            else ""
        )
    elif block_type == "heading_3":
        return (
            "4+" + block["heading_3"]["rich_text"][0]["plain_text"]
            if block["heading_3"]["rich_text"]
            else ""
        )
    elif block_type == "child_page":
        return f"Sub-page: {block['child_page']['title']}"
    else:
        return f"Other: {block}"
    return ""


def get_all_block_children(notion, page_id):
    all_results = []
    start_cursor = None

    while True:
        response = notion.blocks.children.list(
            block_id=page_id, start_cursor=start_cursor
        )

        all_results.extend(response["results"])

        if not response["has_more"]:
            break

        start_cursor = response["next_cursor"]

    return all_results


def get_all_block_content(page_id):
    # blocks = notion.blocks.children.list(block_id=page_id)
    blocks = get_all_block_children(notion, page_id)
    page_content = []
    for block in blocks:
        content = get_block_content(block)
        if content:
            page_content.append(content)
    full_content = "\n".join(page_content)
    return full_content


for k, v in d.items():
    page_id = k
    all_content = get_all_block_content(page_id)
    for i, c in enumerate(all_content.split("\n")):
        print(i, c)
    lines = len(all_content.split("\n"))
    print(lines)
    print(v)
    s.gauge(f"notion.page.lines.{v}", lines)
