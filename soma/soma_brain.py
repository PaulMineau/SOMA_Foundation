"""SOMA Brain — main orchestration loop.

Runs five brain modules in sequence every 30 seconds:
1. Interoception (body signal)
2. Thalamus (routing)
3. Amygdala (affect)
4. Hippocampus (memory)
5. Prefrontal cortex (reasoning)

Each module produces a typed embedding. The affective space merger
combines them into a unified state. Everything persists to LanceDB.

Usage:
    python -m soma.soma_brain
    python -m soma.soma_brain --cycle-seconds 60
    python -m soma.soma_brain --single  # run one cycle and exit
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime

from soma.brain.affective_space import AffectiveSpaceMerger
from soma.brain.amygdala import process as amygdala_process
from soma.brain.embeddings import AffectiveEmbedding, PFCOutput
from soma.brain.hippocampus import HippocampusModule
from soma.brain.interoception import InteroceptionModule
from soma.brain.prefrontal import PrefrontalModule
from soma.brain.state_bus import StateBus
from soma.brain.thalamus import route as thalamus_route
from soma.brain.visual import describe as visual_describe

logger = logging.getLogger(__name__)

DEFAULT_CYCLE_SECONDS = int(os.environ.get("BRAIN_CYCLE_SECONDS", "30"))


class SOMABrain:
    """The brain orchestration loop."""

    def __init__(self, cycle_seconds: int = DEFAULT_CYCLE_SECONDS) -> None:
        self.cycle_seconds = cycle_seconds
        self.interoception = InteroceptionModule()
        self.hippocampus = HippocampusModule()
        self.merger = AffectiveSpaceMerger()
        self.prefrontal = PrefrontalModule()
        self.state_bus = StateBus()
        self.cycle_count = 0

    async def run_cycle(
        self,
        semantic_context: str = "",
        query: str | None = None,
    ) -> PFCOutput:
        """Run one complete brain cycle."""
        cycle_start = datetime.now()
        self.cycle_count += 1

        logger.info("=== Brain cycle %d starting ===", self.cycle_count)

        # 1. Interoception — read body signal
        somatic = self.interoception.process()
        await self.state_bus.publish("interoception", somatic)
        logger.info("Interoception: RMSSD=%.0f RHR=%.0f load=%.2f",
                     somatic.rmssd, somatic.rhr, somatic.load)

        # 2. Visual (stub — returns empty string)
        visual_desc = await visual_describe()

        # 3. Thalamus — route signals
        routing = await thalamus_route(somatic, semantic_context, visual_desc)
        await self.state_bus.publish("thalamus", routing)
        logger.info("Thalamus: %s (low_road=%s)", routing.signal_classification, routing.low_road_flag)

        # 4. Amygdala — classify affect
        affect = await amygdala_process(somatic, routing, semantic_context)
        await self.state_bus.publish("amygdala", affect)
        logger.info("Amygdala: valence=%.2f arousal=%.2f drive=%s",
                     affect.valence, affect.arousal, affect.dominant_drive)

        # 5. Hippocampus — store + retrieve in parallel
        store_task = asyncio.create_task(
            self.hippocampus.encode_and_store(somatic, affect, semantic_context)
        )
        memory = await self.hippocampus.retrieve(query_text=somatic.description)
        await store_task
        await self.state_bus.publish("hippocampus", memory)
        logger.info("Hippocampus: %d similar moments, pattern=%s",
                     len(memory.similar_moments), memory.pattern_note)

        # 6. Merge into unified affective embedding
        embedding = self.merger.merge(somatic, affect, memory, routing)
        await self.state_bus.publish("affective_embedding", embedding)

        # 7. Prefrontal cortex — reasoning
        pfc_out = await self.prefrontal.process(embedding, memory, query)
        await self.state_bus.publish("prefrontal", pfc_out)
        logger.info("PFC [%s]: anomaly=%s", pfc_out.model_used, pfc_out.anomaly_flag)

        # 8. Cycle metadata
        cycle_ms = (datetime.now() - cycle_start).total_seconds() * 1000
        await self.state_bus.publish("cycle_meta", {
            "cycle": self.cycle_count,
            "duration_ms": round(cycle_ms),
            "model_used": pfc_out.model_used,
            "timestamp": cycle_start.isoformat(),
        })
        logger.info("Cycle %d complete in %.0fms", self.cycle_count, cycle_ms)

        return pfc_out

    async def run(self) -> None:
        """Run the brain loop continuously."""
        logger.info("SOMA Brain starting (cycle every %ds)", self.cycle_seconds)
        while True:
            try:
                pfc_out = await self.run_cycle()
                # Print output for terminal visibility
                if pfc_out.recommendation:
                    print(f"\n  SOMA: {pfc_out.recommendation}")
                if pfc_out.question_for_patient:
                    print(f"  SOMA asks: {pfc_out.question_for_patient}")
                if pfc_out.anomaly_flag:
                    print(f"  ANOMALY: {pfc_out.anomaly_description}")
            except Exception:
                logger.exception("Brain cycle failed")
            await asyncio.sleep(self.cycle_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="SOMA Brain — neural simulation loop")
    parser.add_argument("--cycle-seconds", type=int, default=DEFAULT_CYCLE_SECONDS,
                        help="Seconds between brain cycles")
    parser.add_argument("--single", action="store_true",
                        help="Run one cycle and exit")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    brain = SOMABrain(cycle_seconds=args.cycle_seconds)

    if args.single:
        pfc = asyncio.run(brain.run_cycle())
        print(f"\nCycle complete. Model: {pfc.model_used}")
        if pfc.recommendation:
            print(f"Recommendation: {pfc.recommendation}")
        if pfc.anomaly_flag:
            print(f"Anomaly: {pfc.anomaly_description}")
    else:
        print(f"\nSOMA Brain starting — cycle every {args.cycle_seconds}s")
        print("Press Ctrl+C to stop.\n")
        asyncio.run(brain.run())


if __name__ == "__main__":
    main()
