"""C4 perplexity task implementations."""

from collections.abc import Iterator
from typing import Any
from unicode_segmentation_rs import split_word_bound_indices

from olmo_eval.common.metrics import CorpusPerplexityMetric
from olmo_eval.common.types import (
    Instance,
    LMRequest,
    RequestType,
    Split,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant


class CodeFreshBase(Task):
    """Base class for CodeFresh perplexity tasks."""

    split = Split.TRAIN

    @property
    def request_type(self) -> RequestType:
        if self.config.formatter is not None:
            return self.config.formatter.request_type
        return RequestType.LOGLIKELIHOOD

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the dataset."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self.config.get_data_source()
            for doc in loader.load(source):
                self._instances_cache.append(self.process_doc(doc))
        yield from self._instances_cache

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert a dataset document to an Instance."""
        text = doc["file_contents"].strip()

        return Instance(
            question="",  # Context
            gold_answer=text,  # The text we score as the "continuation"
            metadata={
                "id": index,
                "num_chars": len(text),
                "num_words": len(split_word_bound_indices(text)),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())
        gold = instance.gold_answer
        continuations = (gold,) if gold is not None else None
        return LMRequest(
            request_type=self.request_type,
            prompt=instance.question,
            continuations=continuations,
        )


@register("code_fresh_file_blade")
class CodeFreshFileBlade(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="Blade"
    )


@register("code_fresh_file_c")
class CodeFreshFileC(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="C")


@register("code_fresh_file_csharp")
class CodeFreshFileCSharp(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="C#")


@register("code_fresh_file_cpp")
class CodeFreshFileCpp(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="C++"
    )


@register("code_fresh_file_css")
class CodeFreshFileCss(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="CSS"
    )


@register("code_fresh_file_clojure")
class CodeFreshFileClojure(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="Clojure"
    )


@register("code_fresh_file_common_lisp")
class CodeFreshFileCommonLisp(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="Common Lisp"
    )


@register("code_fresh_file_dart")
class CodeFreshFileDart(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="Dart"
    )


@register("code_fresh_file_erlang")
class CodeFreshFileErlang(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="Erlang"
    )


@register("code_fresh_file_fortran")
class CodeFreshFileFortran(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="Fortran"
    )


@register("code_fresh_file_go")
class CodeFreshFileGo(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="Go")


@register("code_fresh_file_html")
class CodeFreshFileHTML(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="HTML"
    )


@register("code_fresh_file_haskell")
class CodeFreshFileHaskell(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="Haskell"
    )


@register("code_fresh_file_java")
class CodeFreshFileJava(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="Java"
    )


@register("code_fresh_file_java_server_page")
class CodeFreshFileJavaServerPage(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="Java Server Page"
    )


@register("code_fresh_file_javascript")
class CodeFreshFileJavaScript(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="JavaScript"
    )


@register("code_fresh_file_julia")
class CodeFreshFileJulia(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="Julia"
    )


@register("code_fresh_file_kotlin")
class CodeFreshFileKotlin(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="Kotlin"
    )


@register("code_fresh_file_lua")
class CodeFreshFileLua(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="Lua"
    )


@register("code_fresh_file_markdown")
class CodeFreshFileMarkdown(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="Markdown"
    )


@register("code_fresh_file_mathematica")
class CodeFreshFileMathematica(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="Mathematica"
    )


@register("code_fresh_file_matlab")
class CodeFreshFileMatlab(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="Matlab"
    )


@register("code_fresh_file_ocaml")
class CodeFreshFileOCaml(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="OCaml"
    )


@register("code_fresh_file_obj_c")
class CodeFreshFileObjectiveC(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="Objective-C"
    )


@register("code_fresh_file_obj_cpp")
class CodeFreshFileObjCpp(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="Objective-C++"
    )


@register("code_fresh_file_php")
class CodeFreshFilePHP(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="PHP"
    )


@register("code_fresh_file_perl")
class CodeFreshFilePerl(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="Perl"
    )


@register("code_fresh_file_powershell")
class CodeFreshFilePowerShell(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="PowerShell"
    )


@register("code_fresh_file_python")
class CodeFreshFilePython(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="Python"
    )


@register("code_fresh_file_ruby")
class CodeFreshFileRuby(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="Ruby"
    )


@register("code_fresh_file_rust")
class CodeFreshFileRust(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="Rust"
    )


@register("code_fresh_file_scala")
class CodeFreshFileScala(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="Scala"
    )


@register("code_fresh_file_scheme")
class CodeFreshFileScheme(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="Scheme"
    )


@register("code_fresh_file_swift")
class CodeFreshFileSwift(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="Swift"
    )


@register("code_fresh_file_tcl")
class CodeFreshFileTcl(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="Tcl"
    )


@register("code_fresh_file_tex")
class CodeFreshFileTeX(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="TeX"
    )


@register("code_fresh_file_typescript")
class CodeFreshFileTypeScript(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="TypeScript"
    )


@register("code_fresh_file_vue")
class CodeFreshFileVue(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="Vue"
    )


@register("code_fresh_file_rest")
class CodeFreshFileRestructuredText(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="reStructuredText"
    )


@register("code_fresh_file_sys_verilog")
class CodeFreshFileSystemVerilog(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="systemverilog"
    )


@register("code_fresh_file_verilog")
class CodeFreshFileVerilog(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="verilog"
    )


@register("code_fresh_file_vhdl")
class CodeFreshFileVhdl(CodeFreshBase):
    """CodeFresh file perplexity task."""

    data_source = DataSource(
        path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file", subset="vhdl"
    )


# =============================================================================
# Variant Registrations
# =============================================================================

register_variant(
    "code_fresh_file_blade",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_c",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_csharp",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_cpp",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_css",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_clojure",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_common_lisp",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_dart",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_erlang",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_fortran",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_go",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_html",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_haskell",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_java",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_java_server_page",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_javascript",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_julia",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_kotlin",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_lua",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_markdown",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_mathematica",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_matlab",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_ocaml",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_obj_c",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_obj_cpp",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_php",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_perl",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_powershell",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_python",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_ruby",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_rust",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_scala",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_scheme",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_swift",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_tcl",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_tex",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_typescript",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_vue",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_rest",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_sys_verilog",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_verilog",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
register_variant(
    "code_fresh_file_vhdl",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
