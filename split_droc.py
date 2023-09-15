import cassis
from tqdm import tqdm
import os
import argparse
import sys
from os import path

FULL_TEXTS = ["DerBlondeEckbert.txt.xmi", "Fontane,-Theodor__Effi Briest.xml.xmi.xmi"]

DEV = [
    "Ahlefeld,-Charlotte-von__Marie Müller.xmi.xmi.xmi",
    "Arnim,-Bettina-von__Clemens Brentanos Frühlingskranz.xmi.xmi.xmi",
    "Aston,-Louise__Lydia.xmi.xmi.xmi",
    "Auerbach,-Berthold__Barfüßele.xmi.xmi.xmi",
    "Beer,-Johann__Das Narrenspital.xmi.xmi.xmi",
    "Braun,-Lily__Lebenssucher.xmi.xmi.xmi",
    "Crébillon,-Claude-Prosper-Jolyot-de__Der Schaumlöffel.xmi.xmi.xmi",
    "Ebner-Eschenbach,-Marie-von__Agave.xmi.xmi.xmi",
    "Ehrmann,-Marianne__Amalie. Eine wahre Geschichte in Briefen.xml.xmi.xmi.xmi",
    "Frapan,-Ilse__Wir Frauen haben kein Vaterland.xmi.xmi.xmi",
    "Kafka,-Franz__Amerika.xml.xmi.xmi.txt.xmi.xmi.xmi",
    "Ludwig,-Otto__Die Heiteretei und ihr Widerspiel.xml.xmi.xmi.txt.xmi.xmi.xmi",
    "Reventlow,-Franziska-Gräfin-zu__Ellen Olestjerne.xml.xmi.xmi.txt.xmi.xmi.xmi",
    "Unger,-Friederike-Helene__Bekenntnisse einer schönen Seele.xml.xmi.xmi.txt.xmi.xmi.xmi",
]

TEST = [
    "Ahlefeld,-Charlotte-von__Erna.xmi.xmi.xmi",
    "Anonym__Schwester Monika.xmi.xmi.xmi",
    "Arnim-Bettina-von__Goethes-Briefwechsel-mit-einem-Kinde.xmi.xmi",
    "Aston,-Louise__Revolution und Contrerevolution.xmi.xmi.xmi",
    "Auerbach,-Berthold__Der Lehnhold.xmi.xmi.xmi",
    "Balzac,-Honoré-de__Vater Goriot.xmi.xmi.xmi",
    "Bierbaum,-Otto-Julius__Stilpe. Ein Roman aus der Froschperspektive.xmi.xmi.xmi",
    "Bruckbräu,-Friedrich-Wilhelm__Mittheilungen aus den geheimen Memoiren einer deutschen Sängerin.xmi.xmi.xmi",
    "Duncker,-Dora__Großstadt.xmi.xmi.xmi",
    "Fischer,-Caroline-Auguste__Gustavs Verirrungen.xmi.xmi.xmi",
    "Fontane,-Theodor__Quitt.xmi.xmi.xmi",
    "Fontane,-Theodor__Stine.xmi.xmi.xmi",
    "Franzos,-Karl-Emil__Der Pojaz.xmi.xmi.xmi",
    "Klabund__Bracke.xml.xmi.xmi.txt.xmi.xmi.xmi",
    "May,-Karl__Auf fremden Pfaden.xml.xmi.xmi.txt.xmi.xmi.xmi",
    "Pichler,-Karoline__Agathocles.xml.xmi.xmi.txt.xmi.xmi.xmi",
    "Sack,-Gustav__Paralyse.xml.xmi.xmi.txt.xmi.xmi.xmi",
    "Verne,-Jules__Zwanzigtausend Meilen unter'm Meer.xml.xmi.xmi.txt.xmi.xmi.xmi",
]

TRAIN = [
    "Raabe,-Wilhelm__Der Schüdderump.xml.xmi.xmi.txt.xmi.xmi.xmi",
    "Stifter,-Adalbert__Zwei Schwestern.xml.xmi.xmi.txt.xmi.xmi.xmi",
    "Bleibtreu,-Karl__Größenwahn.xmi.xmi.xmi",
    "Willkomm,-Ernst-Adolf__Die Europamüden.xml.xmi.xmi.txt.xmi.xmi.xmi",
    "François,-Louise-von__Der Katzenjunker.xmi.xmi.xmi",
    "Janitschek,-Maria__Die Amazonenschlacht.xmi.xmi.xmi",
    "Fontane,-Theodor__Der Stechlin.xmi.xmi.xmi",
    "Boy-Ed,-Ida__Fanny Förster.xmi.xmi.xmi",
    "Ebner-Eschenbach,-Marie-von__Das Gemeindekind.xmi.xmi.xmi",
    "Bahr,-Hermann__Die gute Schule.xmi.xmi.xmi",
    "Meysenbug,-Malwida-Freiin-von__Unerfüllt.xml.xmi.xmi.txt.xmi.xmi.xmi",
    "Balzac,-Honoré-de__Glanz und Elend der Kurtisanen.xmi.xmi.xmi",
    "Zola,-Émile__Nana.xml.xmi.xmi.txt.xmi.xmi.xmi",
    "Swift,-Jonathan__Gullivers Reisen.xml.xmi.xmi.txt.xmi.xmi.xmi",
    "Aston,-Louise__Aus dem Leben einer Frau.xmi.xmi.xmi",
    "Brontë,-Charlotte__Jane Eyre.xmi.xmi.xmi",
    "Dickens,-Charles__Oliver Twist oder Der Weg eines Fürsorgezöglings.xmi.xmi.xmi",
    "Arnim,-Ludwig-Achim-von__Isabella von Ägypten.xmi.xmi.xmi",
    "Fouqué,-Caroline-de-la-Motte__Resignation.xmi.xmi.xmi",
    "Fontane-Theodor__Cécile.xmi.xmi",
    "Flaubert,-Gustave__Madame Bovary.xmi.xmi.xmi",
    "Meinhold,-Wilhelm__Die Bernsteinhexe.xml.xmi.xmi.txt.xmi.xmi.xmi",
    "Arnim,-Bettina-von__Die Günderode.xmi.xmi.xmi",
    "Gotthelf,-Jeremias__Uli der Pächter.xmi.xmi.xmi",
    "Spielhagen,-Friedrich__Problematische Naturen. Zweite Abtheilung (Durch Nacht zum Licht).xml.xmi.xmi.txt.xmi.xmi.xmi",
    "Lewald,-Fanny__Jenny.xml.xmi.xmi.txt.xmi.xmi.xmi",
    "Raabe,-Wilhelm__Stopfkuchen. Eine See- und Mordgeschichte.xml.xmi.xmi.txt.xmi.xmi.xmi",
    "Argens,-Jean-Baptiste-Boyer,-Marquis-d'__Die philosophische Therese.xmi.xmi.xmi",
    "Wieland,-Christoph-Martin__Peregrinus Proteus.xml.xmi.xmi.txt.xmi.xmi.xmi",
    "Mundt,-Theodor__Madonna. Unterhaltungen mit einer Heiligen.xml.xmi.xmi.txt.xmi.xmi.xmi",
    "Janitschek,-Maria__Einer Mutter Sieg.xmi.xmi.xmi",
    "Beer,-Johann__Teutsche Winter-Nächte.xmi.xmi.xmi",
    "Knigge,-Adolph-Freiherr-von__Josephs von Wurmbrands ... politisches Glaubensbekenntnis.xml.xmi.xmi.txt.xmi.xmi.xmi",
    "Auerbach,-Berthold__7. Ivo_ der Hajrle.xmi.xmi.xmi",
    "Wassermann,-Jakob__Die Juden von Zirndorf.xml.xmi.xmi.txt.xmi.xmi.xmi",
    "Arnim,-Ludwig-Achim-von__Erster Band.xmi.xmi.xmi",
    "Ball,-Hugo__Flammetti.xmi.xmi.xmi",
    "Christ,-Lena__Die Rumplhanni.xmi.xmi.xmi",
    "Otto,-Louise__Zweiter Band.xml.xmi.xmi.txt.xmi.xmi.xmi",
    "Schopenhauer,-Johanna__Richard Wood.xml.xmi.xmi.txt.xmi.xmi.xmi",
    "Auerbach,-Berthold__2. Die Frau Professorin.xmi.xmi.xmi",
    "Auerbach,-Berthold__Lucifer.xmi.xmi.xmi",
    "Fontane,-Theodor__Schach von Wuthenow.xmi.xmi.xmi",
    "Fontane,-Theodor__Grete Minde.xmi.xmi.xmi",
    "Dohm,-Hedwig__Wie Frauen werden.xmi.xmi.xmi",
    "Ebner-Eschenbach,-Marie-von__Lotti, die Uhrmacherin.xmi.xmi.xmi",
    "Tolstoj,-Lev-Nikolaevic__Anna Karenina.xml.xmi.xmi.txt.xmi.xmi.xmi",
    "Ball,-Hugo__Tenderenda der Phantast.xmi.xmi.xmi",
    "Dauthendey,-Max__Lingam.xmi.xmi.xmi",
    "Alexis,-Willibald__Die Hosen des Herrn von Bredow.xmi.xmi.xmi",
    "Ebner-Eschenbach,-Marie-von__Bozena.xmi.xmi.xmi",
    "Boy-Ed,-Ida__Vor der Ehe.xmi.xmi.xmi",
    "Spyri,-Johanna__Heidi kann brauchen_ was es gelernt hat.xml.xmi.xmi.txt.xmi.xmi.xmi",
    "Raabe,-Wilhelm__Die Leute aus dem Walde_ ihre Sterne_ Wege und Schicksale.xml.xmi.xmi.txt.xmi.xmi.xmi",
    "Fontane,-Theodor__Mathilde Möhring.xmi.xmi.xmi",
    "May,-Karl__Im Reiche des silbernen Löwen IV.xml.xmi.xmi.txt.xmi.xmi.xmi",
    "Reventlow,-Franziska-Gräfin-zu__Herrn Dames Aufzeichnungen.xml.xmi.xmi.txt.xmi.xmi.xmi",
    "Gutzkow,-Karl__Wally_ die Zweiflerin.xmi.xmi.xmi",
]

BOOKS = TRAIN + DEV + TEST

def get_typesystem(typesystem_path):
    with open(typesystem_path, "rb") as f:
        typesystem = cassis.load_typesystem(f)
        # We have to add some types that apparently are not in the XML :/
        ep = typesystem.create_type("de.uniwue.mk.kall.Erzaehlpassage")
        sa = typesystem.create_type("de.uniwue.mk.kall.Sprechakt")
        typesystem.add_feature(type_=sa, name="Aufbau", rangeTypeName="uima.cas.String")
        typesystem.create_type("de.uniwue.mk.kall.SprechaktText")
        dialogue = typesystem.create_type("de.uniwue.mk.kall.Dialog")
        typesystem.add_feature(type_=dialogue, name="Sprechakte", rangeTypeName="uima.cas.Integer")
    return typesystem

def remove_three_letter_extensions(name):
    while name.endswith(".txt") or name.endswith(".xmi") or name.endswith(".xml"):
        name = name[:-4]
    return name

def get_file_and_split(filename, individual_files, out_files, dirname):
    if not individual_files:
        assert len(out_files) == 3
    in_basename = path.basename(filename)
    if in_basename in TRAIN:
        split = "train"
    elif in_basename in DEV:
        split = "development"
    elif in_basename in TEST:
        split = "test"
    else:
        raise ValueError("Invalid split")
    basename = remove_three_letter_extensions(in_basename)
    if individual_files:
        out_name = path.join(dirname, f"{split}", basename + ".gold_conll")
        return open(out_name, "w"), split
    else:
        return out_files[split], split

def convert_library(in_path, output_file_name, typesystem_path=None, split_paragraphs=False, max_length=None, individual_files=False):
    typesystem = get_typesystem(typesystem_path)
    TrainingDoc = typesystem.create_type("de.uhh.inf.lt.TrainingDocument")
    file_names = []
    for file_name in BOOKS:
        file_names.append(os.path.join(in_path, file_name))
    dirname = path.dirname(output_file_name)
    basename = path.basename(output_file_name)
    if individual_files:
        out_files = None
        for split in ["train", "development", "test"]:
            os.makedirs(path.join(dirname, split), exist_ok=True)
    else:
        out_files = {split : open(path.join(dirname, f"{split}." + basename), "w") for split in ["train", "development", "test"]}
    for doc_num, filename in tqdm(enumerate(file_names), desc="Processing documents"):
        in_file = open(filename, "rb")
        doc_id = filename.split("/")[-1].split(".")[0].replace(" ", "_") + f".{doc_num:05d}"
        cas = cassis.load_cas_from_xmi(in_file, typesystem=typesystem)
        output_file, output_split = get_file_and_split(filename, individual_files, out_files, dirname)
        if split_paragraphs:
            write_paragraphs_as_docs(cas, output_file, doc_id, TrainingDoc)
        else:
            write_document(cas, output_file, doc_id, max_length=max_length if output_split == "train" else None)

def write_paragraphs_as_docs(cas, output_file, doc_id, TrainingDoc):
    paragraph_num = 1
    previous_end = 0
    for paragraph in cas.select("de.uniwue.kalimachos.coref.type.Paragraph"):
        # This is effectively a minimum length
        if paragraph.end - previous_end < (384 * 3):
            continue
        else:
            custom_paragraph = TrainingDoc()
            custom_paragraph.begin = previous_end
            custom_paragraph.end = paragraph.end
            previous_end = paragraph.end
        if len(cas.select_covered("de.uniwue.kalimachos.coref.type.Sentence", custom_paragraph)) > 0:
            write_document(
                cas,
                output_file,
                doc_id + f"_{paragraph_num:03d}",
                paragraph=paragraph,
            )
            paragraph_num += 1


def write_document(cas, output_file, doc_id, paragraph=None, max_length=None):
    if not paragraph:
        sentences = list(cas.select("de.uniwue.kalimachos.coref.type.Sentence"))
    else:
        sentences = list(cas.select_covered("de.uniwue.kalimachos.coref.type.Sentence", paragraph))
    if len(sentences) == 0:
        return
    if max_length is None:
        output_file.write(f"#begin document {doc_id}\n")
    active_entities = set()
    words = 0
    sub_doc = 0
    for sentence_number, sentence in enumerate(sentences, 1):
        tokens = cas.select_covered("de.uniwue.kalimachos.coref.type.POS", sentence)
        sentence_rows = []
        for word_num, token in enumerate(tokens, 1):
            lemma = token.Lemma
            coref = "-"
            pos = "-"
            word = token.get_covered_text()
            entities = list(cas.select_covering("de.uniwue.kalimachos.coref.type.NamedEntity", token))
            dependency = list(cas.select_covering("de.uniwue.kalimachos.coref.type.DependencyParse", token))
            assert len(dependency) <= 1
            if len(dependency) == 1:
                head = dependency[0].WordNumber
                dep_rel = dependency[0].DependencyRelation
            else:
                head = "-"
                dep_rel = "-"
            starting_entities = set()
            closing_entities = set()
            for e in entities:
                if e.begin <= token.begin and e not in active_entities:
                    starting_entities.add(e)
                    active_entities.add(e)
            for e in active_entities:
                if e.end <= token.end:
                    closing_entities.add(e)
            coref = []
            for single_token_entity in set(e.ID for e in starting_entities & closing_entities):
                coref.append(f"({single_token_entity})")
            for starting_entity in set(e.ID for e in starting_entities - closing_entities):
                coref.append(f"({starting_entity}")
            for ending_entity in set(e.ID for e in closing_entities - starting_entities):
                coref.append(f"{ending_entity})")
            row = [
                str(word_num),
                word,
                lemma or "-",
                "-",
                pos,
                "-",
                "-",
                "-",
                head,
                "-",
                dep_rel,
                "-",
                "-",
                "-",
                "-",
                "-",
                "|".join(coref) if len(coref) > 0 else "-",
            ]
            sentence_rows.append(row)
            active_entities -= closing_entities
            words += 1
        if max_length:
            if words == len(sentence_rows): # We are in the first iteration
                output_file.write(f"#begin document {doc_id}_{sub_doc:03d}\n")
            if words >= max_length:
                output_file.write("#end document\n")
                sub_doc += 1
                output_file.write(f"#begin document {doc_id}_{sub_doc:03d}\n")
                words = len(sentence_rows)
            for row in sentence_rows:
                row[0] += f"_{sub_doc:03d}"
                output_file.write(
                    "\t".join(row) + "\n"
                )
        if max_length is None:
            for row in sentence_rows:
                output_file.write(
                    "\t".join(row) + "\n"
                )
        if sentence_number != len(sentences):
            output_file.write("\n")  # newline after each sentence but the last one
    output_file.write("#end document\n")
    if len(active_entities) != 0:
        raise Exception(f"Open entities at the end of document {doc_id}: {active_entities}")


def convert_full_texts(xmi_path, base_path, typesystem_path):
    full_paths = [(os.path.join(xmi_path, text_name), remove_three_letter_extensions(text_name)) for text_name in FULL_TEXTS]
    typesystem = get_typesystem(typesystem_path)
    for file_path, text_name in full_paths:
        in_file = open(file_path, "rb")
        cas = cassis.load_cas_from_xmi(in_file, typesystem=typesystem)
        os.makedirs(base_path, exist_ok=True)
        output_file = open(os.path.join(base_path, text_name + ".gold_conll"), "w")
        write_document(cas, output_file, text_name)


def main():
    parser = argparse.ArgumentParser("Convert DROC files to ConLL-2012 format.")
    parser.add_argument("xmi_path", type=str)
    parser.add_argument("output_conll_prefix", type=str)
    parser.add_argument("--full-texts", help="Switch to full text conversion mode", action="store_true")
    parser.add_argument("--type-system-xml", help="Path to typesystem XML file", default="CorefTypeSystem.xml", type=str)
    parser.add_argument("--split-paragraphs", help="Split paragraphs into individual documents.", action="store_true")
    parser.add_argument("--max-length", help="Split training into documents of maximum length (always using the previous sentence boundary), only done for training set.", type=int, default=None)
    parser.add_argument("--individual-files", help="Individual file for each document", action="store_true")
    args = parser.parse_args()

    if args.max_length and args.split_paragraphs:
        print("--max-length and --split-paragraphs are mutually exclusive")
        sys.exit(1)

    if args.split_paragraphs:
        print("Splitting documents into paragraphs!")
    if args.full_texts:
        convert_full_texts(args.xmi_path, args.output_conll_prefix, args.type_system_xml)
    else:
        convert_library(args.xmi_path, args.output_conll_prefix, args.type_system_xml, args.split_paragraphs, max_length=args.max_length, individual_files=args.individual_files)

if __name__ == "__main__":
    main()
