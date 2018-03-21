import TransformOp from "./TransformOp";

export default class AtanhOp extends TransformOp {

  body(a: number, b?: number): number {
    return Math.atanh(a);
  }

}