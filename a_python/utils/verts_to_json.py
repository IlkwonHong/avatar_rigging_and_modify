import json
import pickle as pkl


if __name__ == "__main__":
    with open('/Users/ik/Desktop/zepeto/a_python/to_py_mesh_.pkl', 'rb') as f:
        vertices = pkl.load(f)

    json_save_path = './io_files/vertices_out.json'
    # read_path = './data/for_json.json'
    # with open(read_path, 'rb') as f:
    #     data = json.load(f)

    json_vertices = {}
    json_vertices["deformerWeight"] = {}
    json_vertices["deformerWeight"]["shapes"] = []
    json_vertices["deformerWeight"]["shapes"].append(
        {
            "name": "baseShape",
            "group": 0,
            "stride": 3,
            "size": 9067,
            "max": 9067,
            "points" : []
        }
    )

    for _idx, vertex in enumerate(vertices) :
        json_vertices["deformerWeight"]["shapes"][0]["points"].append({
            "index": _idx,
            "value": [
                vertex[0],
                vertex[1],
                vertex[2]
                ]
        })


    # json_vertices["deformerWeight"]["shapes"][0]["points"].append({
    #     "index": 0,
    #     "value": [
    #         -3.933340072631836,
    #         99.70133972167969,
    #         7.72953987121582
    #     ]
    # })

    with open(json_save_path, 'w') as f:
        json.dump(json_vertices, f, indent=4, sort_keys=True)


    # json_vertices["deformerWeight"]["shapes"][0]["name"] = "baseShape"
    # json_vertices["deformerWeight"]["shapes"][0]["group"] = 0
    # json_vertices["deformerWeight"]["shapes"][0]["stride"] = 3
    # json_vertices["deformerWeight"]["shapes"][0]["size"] = 9067
    # json_vertices["deformerWeight"]["shapes"][0]["max"] = 9067




    # data["deformerWeight"]["shapes"][0]["name"] = "baseShape"
    # data["deformerWeight"]["shapes"][0]["group"] = 0
    # data["deformerWeight"]["shapes"][0]["stride"] = 3
    # data["deformerWeight"]["shapes"][0]["size"] = 9067
    # data["deformerWeight"]["shapes"][0]["max"] = 9067

    # data["deformerWeight"]["shapes"][0]["points"][0]["index"]
    # data["deformerWeight"]["shapes"][0]["points"][0]["value"]


