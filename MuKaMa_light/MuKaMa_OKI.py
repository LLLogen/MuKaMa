import requests
import networkx as nx
def query_wikidata_and_parse_results(query):
    url = 'https://www.wikidata.org/w/api.php'
    params = {
        'action': 'wbsearchentities',
        'language': 'en',
        'format': 'json',
        'search': query
    }
    
    # 发送请求并解析结果
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # 
        results = response.json().get('search', [])
        
        # 解析并保存结果
        parsed_results = {}
        for result in results:
            item_id = result['id']
            label = result.get('label', 'No label available')
            description = result.get('description', 'No description available')
            
            parsed_results[item_id] = {'label': label, 'description': description}
        
        return parsed_results
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}


def enrich_with_tagme(parsed_results, tagme_token):
    tagme_url = "https://tagme.d4science.org/tagme/tag"
    headers = {"Authorization": "Bearer " + tagme_token}

    for item_id, info in parsed_results.items():
        description = info.get('description', '')
        response = requests.get(tagme_url, headers=headers, params={'text': description})
        if response.status_code == 200:
            annotations = response.json().get('annotations', [])
            entities = []
            for annotation in annotations:
                if 'title' in annotation:
                    entities.append(annotation['title'])
            info['entities'] = entities
        else:
            print(f"Error with TagMe API for item {item_id}")

    return parsed_results

def construct_knowledge_graph(enriched_results):
    """
    根据富含TagMe实体的字典构造知识图谱。

    :param enriched_results: enrich_with_tagme函数返回的富含实体的字典。
    :return: 一个networkx图，代表知识图谱。
    """
    G = nx.Graph()

    for item_id, info in enriched_results.items():
        label = info.get('label', 'Unknown')
        # 添加标签节点（如果尚未存在）
        if label not in G:
            G.add_node(label, type='label')

        entities = info.get('entities', [])
        for entity in entities:
            # 添加实体节点（如果尚未存在）
            if entity not in G:
                G.add_node(entity, type='entity')
            # 添加标签和实体之间的边
            G.add_edge(label, entity)

    return G