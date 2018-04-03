import ComparisonOp from "./ComparisonOp";

export default class LessEqualOp extends ComparisonOp {

  body(a: number, b?: number): number {
    return a <= b ? 1 : 0;
  }

}