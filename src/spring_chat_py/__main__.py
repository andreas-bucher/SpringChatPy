'''
Created on 28 Jan 2026

@author: ante
'''
import sys, yaml
import logging.config
from spring_chat_py import verify_tool_descriptions
from spring_chat_py.extract import pdf_outlines
from spring_chat_py.embeddings import embed_chunks

log = logging.getLogger(__name__)

with open("logging.yaml") as f:
    config = yaml.safe_load(f)
logging.config.dictConfig(config)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv

    if len(argv) != 2:
        log.info("Usage: python script.py <verify_tool_desc>")
        return 1

    command = argv[1]

    match command:
        case "verify_tool_desc":
            return verify_tool_descriptions.main(argv)
        case "certs_outline":
            return pdf_outlines.gen_outline("docs/certificates", "rag/outlines/certificates")     
        case "create_embeddings":
            #embed_chunks.create_chunks("docs/designing_ai_products_and_services", "rag/chunks/designing_ai_products_and_services")
            embed_chunks.create_embeddings("docs/designing_ai_products_and_services", "rag/embeddings/designing_ai_products_and_services")
        case _:
            log.error(f"Unknown command: %s", command)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())