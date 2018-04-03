import ComparisonOp from "./ComparisonOp";

export default class GreaterEqualOp extends ComparisonOp {

  body(a: number, b?: number): number {
    return a >= b ? 1 : 0;
  }

}