# cs_assigment

##Query: [“疾病“]诊断的检查项目有哪些？请提供给我全面准确的回答。 ## used to validate the performanace of chatgpt

## used to validate the performance about chatgpt for cypher generation
##Query for cypher generation: [“疾病”]诊断的检查项目有哪些？知识库中存在[“diseases”]和[“checks”]两类实体，他们之间的关系是[“诊断检查”]，请提供给我基于这个知识库查询这个问题的Cypher语句

Cypher: 
Math (d: Disease{name: “疾病”})-[:诊断检查]->(diag: checks)
RETURN diag.name

Ground_truth: XXXXX, XXXX, XXXX

Modified Query: [“疾病“]诊断的检查项目有哪些？这些是我从知识库中找到的供你参考的信息"颅脑CT检查", "颅脑MRI检查", "脑电图检查", "脑脊液乳酸", "磁共振血管造影”。请提供给我全面准确的回答。


## Random sampling and select all nodes to construct such data

疾病-诊断检查, 疾病-症状, 疾病-并发症, 疾病-所属科室, 疾病-治疗方法, 疾病-常用药品

How to evaluate:
1 character match
2 Ask ChatGPT again（example below）
"颅脑CT检查", "颅脑MRI检查", "脑电图检查", "脑脊液乳酸", "磁共振血管造影” 以下描述中是否包含了这些信息，请给出你的判断并且打分，满分100分，每个项目20分。你可以这么说
判断依据：在这里给出你判断的过程和依据
分数：100
