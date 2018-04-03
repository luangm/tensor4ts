import ComparisonOp from "./ComparisonOp";

export default class EqualOp extends ComparisonOp {

  body(a: number, b?: number): number {
    return a === b ? 1 : 0;
  }

}