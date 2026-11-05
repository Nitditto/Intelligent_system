import sys
sys.stdout.reconfigure(encoding="utf-8")
from neo4j import GraphDatabase
import csv

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "diabetes123")

neo4j_ran = False
try:
    driver = GraphDatabase.driver(URI, auth=AUTH)
    driver.verify_connectivity()
    query = """
    MATCH (:DiabetesType {name: 'Đái tháo đường'})-[r:HAS_COMPLICATION]->(c:Complication)
    OPTIONAL MATCH (c)-[:AFFECTS_ORGAN]->(o:Organ)
    RETURN c.name AS complication, o.name AS organ, r.support AS articles
    ORDER BY articles DESC LIMIT 10;
    """
    with driver.session() as s:
        res = list(s.run(query))
        print("RESULT_FROM: Neo4j")
        for r in res:
            print(f"{r['complication']} | {r['organ']} | {r['articles']}")
    driver.close()
    neo4j_ran = True
except Exception as e:
    print(f"Neo4j offline ({e}), falling back to CSV calculation...")

if not neo4j_ran:
    # calculate from processed CSVs: triples.csv + entities.csv
    with open("dataset/processed/triples.csv", encoding="utf-8") as f:
        triples = list(csv.DictReader(f))
    
    # complication to organ map from curated facts in triples.csv
    comp_organ = {}
    for t in triples:
        if t["relation"] == "AFFECTS_ORGAN":
            comp_organ[t["subject"]] = t["object"]

    # find HAS_COMPLICATION for 'Đái tháo đường'
    complications = []
    for t in triples:
        if t["subject"] == "Đái tháo đường" and t["relation"] == "HAS_COMPLICATION":
            complications.append((t["object"], comp_organ.get(t["object"]), int(t["support"])))
    
    complications.sort(key=lambda x: x[2], reverse=True)
    print("RESULT_FROM: CSV")
    for comp, organ, sup in complications[:10]:
        print(f"{comp} | {organ} | {sup}")
