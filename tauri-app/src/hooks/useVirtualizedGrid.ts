import { useRef, useState, useEffect } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";

/** Match Tailwind responsive breakpoints: grid-cols-2 sm:3 md:4 lg:5 xl:6 */
function getColumns(viewportWidth: number): number {
  if (viewportWidth >= 1280) return 6;
  if (viewportWidth >= 1024) return 5;
  if (viewportWidth >= 768) return 4;
  if (viewportWidth >= 640) return 3;
  return 2;
}

export function useVirtualizedGrid(
  itemCount: number,
  estimateRowHeight = 250,
  overscan = 3,
) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [columns, setColumns] = useState(() => getColumns(window.innerWidth));

  useEffect(() => {
    const handler = () => setColumns(getColumns(window.innerWidth));
    window.addEventListener("resize", handler);
    return () => window.removeEventListener("resize", handler);
  }, []);

  const rowCount = Math.ceil(itemCount / columns);

  const virtualizer = useVirtualizer({
    count: rowCount,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => estimateRowHeight,
    overscan,
  });

  // Re-measure when column count changes (row heights change with card width)
  useEffect(() => {
    virtualizer.measure();
  }, [columns]);

  return { scrollRef, virtualizer, columns };
}
