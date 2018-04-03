import ComparisonOp from "./ComparisonOp";

export default class LessOp extends ComparisonOp {

  body(a: number, b?: number): number {
    return a < b ? 1 : 0;
  }

}