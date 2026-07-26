from pathlib import Path
from typing import List, Any

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    JSONLoader,
)

from langchain_community.document_loaders.excel import (
    UnstructuredExcelLoader,
)


def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported documents from the specified directory.

    Supported Formats:
    - PDF
    - DOCX
    - TXT
    - CSV
    - XLSX
    - JSON
    """

    data_path = Path(data_dir).resolve()

    print(f"[INFO] Data Path: {data_path}")

    documents = []

    # ---------------- PDF ---------------- #

    pdf_files = list(data_path.glob("**/*.pdf"))

    print(f"[INFO] Found {len(pdf_files)} PDF file(s).")

    for pdf_file in pdf_files:
        try:
            print(f"[INFO] Loading PDF: {pdf_file}")

            loader = PyPDFLoader(str(pdf_file))

            documents.extend(loader.load())

        except Exception as e:
            print(f"[ERROR] {pdf_file}: {e}")

    # ---------------- TXT ---------------- #

    txt_files = list(data_path.glob("**/*.txt"))

    print(f"[INFO] Found {len(txt_files)} TXT file(s).")

    for txt_file in txt_files:
        try:
            print(f"[INFO] Loading TXT: {txt_file}")

            loader = TextLoader(str(txt_file))

            documents.extend(loader.load())

        except Exception as e:
            print(f"[ERROR] {txt_file}: {e}")

    # ---------------- CSV ---------------- #

    csv_files = list(data_path.glob("**/*.csv"))

    print(f"[INFO] Found {len(csv_files)} CSV file(s).")

    for csv_file in csv_files:
        try:
            print(f"[INFO] Loading CSV: {csv_file}")

            loader = CSVLoader(str(csv_file))

            documents.extend(loader.load())

        except Exception as e:
            print(f"[ERROR] {csv_file}: {e}")

    # ---------------- XLSX ---------------- #

    excel_files = list(data_path.glob("**/*.xlsx"))

    print(f"[INFO] Found {len(excel_files)} Excel file(s).")

    for excel_file in excel_files:
        try:
            print(f"[INFO] Loading Excel: {excel_file}")

            loader = UnstructuredExcelLoader(str(excel_file))

            documents.extend(loader.load())

        except Exception as e:
            print(f"[ERROR] {excel_file}: {e}")

    # ---------------- DOCX ---------------- #

    docx_files = list(data_path.glob("**/*.docx"))

    print(f"[INFO] Found {len(docx_files)} DOCX file(s).")

    for docx_file in docx_files:
        try:
            print(f"[INFO] Loading DOCX: {docx_file}")

            loader = Docx2txtLoader(str(docx_file))

            documents.extend(loader.load())

        except Exception as e:
            print(f"[ERROR] {docx_file}: {e}")

    # ---------------- JSON ---------------- #

    json_files = list(data_path.glob("**/*.json"))

    print(f"[INFO] Found {len(json_files)} JSON file(s).")

    for json_file in json_files:
        try:
            print(f"[INFO] Loading JSON: {json_file}")

            loader = JSONLoader(
                file_path=str(json_file),
                jq_schema=".",
                text_content=False,
            )

            documents.extend(loader.load())

        except Exception as e:
            print(f"[ERROR] {json_file}: {e}")

    print(f"\n[INFO] Total Documents Loaded: {len(documents)}")

    return documents


if __name__ == "__main__":

    docs = load_all_documents("data")

    print(f"\nLoaded {len(docs)} document(s).")

    if docs:
        print("\nExample Document:\n")
        print(docs[0])