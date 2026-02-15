declare module "react-katex" {
  import type * as React from "react";
  import type { KatexOptions } from "katex";

  export type RenderError = (error: Error) => React.ReactNode;

  export interface BaseKatexProps {
    math: string;
    errorColor?: string;
    renderError?: RenderError;
    settings?: KatexOptions;
  }

  export const InlineMath: React.FC<BaseKatexProps & React.HTMLAttributes<HTMLSpanElement>>;
  export const BlockMath: React.FC<BaseKatexProps & React.HTMLAttributes<HTMLDivElement>>;
}
