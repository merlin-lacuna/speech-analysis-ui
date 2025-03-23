import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

interface NameSelectorProps {
  value: string;
  onChange: (value: string) => void;
}

export function NameSelector({ value, onChange }: NameSelectorProps) {
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium">Select user name</label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className="w-[180px]">
          <SelectValue placeholder="Select name" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="Unknown">Unknown</SelectItem>
          <SelectItem value="Merlin">Merlin</SelectItem>
          <SelectItem value="Anda">Anda</SelectItem>
        </SelectContent>
      </Select>
    </div>
  );
}