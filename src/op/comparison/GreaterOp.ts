import ComparisonOp from "./ComparisonOp";

export default class GreaterOp extends ComparisonOp {

  body(a: number, b?: number): number {
    return a > b ? 1 : 0;
  }

}