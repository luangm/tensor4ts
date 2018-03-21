import TransformOp from "./TransformOp";

export default class CosOp extends TransformOp {

  body(a: number, b?: number): number {
    return Math.cos(a);
  }

}