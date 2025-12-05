"""
UT LONGHORN DM - INTERACTIVE D&D GAME (V5 - CLEAN EXTRACTION!)
Better text extraction from LLM-generated campaign documents
"""

import torch
import json
import os
import random
import re
from contextlib import nullcontext
from model import Transformer, ModelArgs

# ============================================================================
# LLM CAMPAIGN GENERATOR WITH BETTER EXTRACTION
# ============================================================================

class CampaignGenerator:
    """Uses your LLM to generate campaign content at game start"""
    
    def __init__(self, checkpoint_path='out_ut_dm/ckpt.pt'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🎲 Loading your trained LLM on {self.device}...")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_args = checkpoint['model_args']
        gptconf = ModelArgs(**model_args)
        self.model = Transformer(gptconf)
        
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)
        self.model_args = model_args
        
        vocab_path = 'data/vocab.json'
        with open(vocab_path, 'r') as f:
            self.char_to_token = json.load(f)
        self.token_to_char = {v: k for k, v in self.char_to_token.items()}
        
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.ctx = nullcontext() if self.device == 'cpu' else torch.amp.autocast(device_type=self.device, dtype=dtype)
        
        print("✅ LLM loaded!\n")
    
    def generate(self, prompt, max_chars=3000):
        """Generate text from your trained model"""
        tokens = [self.char_to_token.get(char, 0) for char in prompt]
        x = torch.tensor(tokens, dtype=torch.long, device=self.device)[None, ...]
        
        generated = []
        eos_token = self.char_to_token.get('<EOS>', -1)
        
        print("   🤖 Generating", end='', flush=True)
        
        with torch.no_grad():
            with self.ctx:
                for i in range(max_chars):
                    if i % 500 == 0:
                        print(".", end='', flush=True)
                    
                    x_cond = x if x.size(1) <= self.model_args['max_seq_len'] else x[:, -self.model_args['max_seq_len']:]
                    
                    logits = self.model(x_cond, x_cond)
                    logits = logits[:, -1, :] / 0.85
                    
                    v, _ = torch.topk(logits, min(200, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                    
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    x = torch.cat((x, next_token), dim=1)
                    generated.append(next_token.item())
                    
                    if next_token.item() == eos_token:
                        break
        
        print(" ✓")
        
        chars = []
        for token in generated:
            if token in self.token_to_char:
                char = self.token_to_char[token]
                if char not in ['<BOS>', '<EOS>']:
                    chars.append(char)
        
        return ''.join(chars)
    
    def clean_text(self, text):
        """Aggressively clean generated text"""
        # Remove markdown formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold**
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *italic*
        text = re.sub(r'#{1,6}\s+', '', text)            # ### headers
        
        # Remove common prefixes
        text = re.sub(r'^Title:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^Type:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^Goal:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^Hook:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^Name:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^Description:\s*', '', text, flags=re.IGNORECASE)
        
        # Remove bullet points and dashes
        text = re.sub(r'^\s*[-•]\s+', '', text, flags=re.MULTILINE)
        
        # Clean up whitespace
        text = ' '.join(text.split())
        text = text.strip()
        
        return text
    
    def extract_sentences(self, text, min_words=5, max_words=40):
        """Extract clean sentences from text"""
        # Split into sentences (roughly)
        sentences = re.split(r'[.!?]+', text)
        
        clean_sentences = []
        for sent in sentences:
            sent = self.clean_text(sent)
            
            # Count words
            words = sent.split()
            if min_words <= len(words) <= max_words:
                # Make sure it starts with capital letter
                if sent and sent[0].isalpha():
                    sent = sent[0].upper() + sent[1:]
                    clean_sentences.append(sent + '.')
        
        return clean_sentences
    
    def generate_campaign_content(self):
        """Generate campaign content and extract usable parts"""
        print("="*70)
        print("🎮 GENERATING CAMPAIGN WITH YOUR LLM")
        print("="*70 + "\n")
        
        content = {
            'factions': [],
            'locations': [],
            'npcs': [],
            'enemies': []
        }
        
        # Generate locations (most useful for descriptions)
        print("🏰 Generating locations...")
        prompt = "### Instruction:\nDescribe interesting locations at the University of Arcane Tejas\n\n### Response:\n"
        text = self.generate(prompt, max_chars=2000)
        sentences = self.extract_sentences(text)
        content['locations'] = sentences[:5] if len(sentences) >= 5 else sentences + ["A mysterious place filled with magic."] * (5 - len(sentences))
        print(f"   ✅ Extracted {len(content['locations'])} location descriptions\n")
        
        # Generate NPCs
        print("👥 Generating NPCs...")
        prompt = "### Instruction:\nCreate NPCs for a university D&D campaign\n\n### Response:\n"
        text = self.generate(prompt, max_chars=2000)
        sentences = self.extract_sentences(text)
        content['npcs'] = sentences[:4] if len(sentences) >= 4 else sentences + ["A mysterious figure watches you."] * (4 - len(sentences))
        print(f"   ✅ Extracted {len(content['npcs'])} NPC descriptions\n")
        
        # Generate enemies
        print("⚔️  Generating enemies...")
        prompt = "### Instruction:\nCreate enemies and monsters for a university setting\n\n### Response:\n"
        text = self.generate(prompt, max_chars=2000)
        sentences = self.extract_sentences(text)
        content['enemies'] = sentences[:4] if len(sentences) >= 4 else sentences + ["A dangerous creature appears."] * (4 - len(sentences))
        print(f"   ✅ Extracted {len(content['enemies'])} enemy descriptions\n")
        
        # Generate quest hooks
        print("📜 Generating quest hooks...")
        prompt = "### Instruction:\nCreate quest hooks for university adventures\n\n### Response:\n"
        text = self.generate(prompt, max_chars=2000)
        sentences = self.extract_sentences(text)
        content['quests'] = sentences[:3] if len(sentences) >= 3 else sentences + ["Something mysterious is happening."] * (3 - len(sentences))
        print(f"   ✅ Extracted {len(content['quests'])} quest hooks\n")
        
        print("="*70)
        print("✨ CAMPAIGN READY!")
        print("="*70 + "\n")
        
        return content

# ============================================================================
# CHARACTER & GAME STATE
# ============================================================================

class Character:
    def __init__(self, name, char_class, race):
        self.name = name
        self.char_class = char_class
        self.race = race
        self.level = 1
        self.hp = 20
        self.max_hp = 20
        self.inventory = ["Adventurer's Pack", "Torch", "Rations"]
        self.gold = 15
        self.xp = 0
        self.attack_bonus = 3 if char_class in ['Fighter', 'Rogue'] else 2
        self.defense = 12 if char_class == 'Fighter' else 11
        
    def __str__(self):
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║  {self.name} - Level {self.level} {self.race} {self.char_class}
╠══════════════════════════════════════════════════════════════════╣
║  HP: {self.hp}/{self.max_hp}  |  Gold: {self.gold}  |  XP: {self.xp}
║  Attack: +{self.attack_bonus}  |  Defense: {self.defense}
╚══════════════════════════════════════════════════════════════════╝
"""

class GameState:
    def __init__(self):
        self.character = None
        self.location_index = 0
        self.enemy_present = False
        self.current_enemy = None
        self.campaign_content = None

# ============================================================================
# DICE & COMBAT
# ============================================================================

def roll_dice(sides=20, bonus=0):
    roll = random.randint(1, sides)
    return roll, roll + bonus

def combat_round(character, enemy_name, enemy_hp, enemy_ac):
    print(f"\n{'─'*70}")
    print(f"💚 Your HP: {character.hp}/{character.max_hp}  |  ❤️  {enemy_name} HP: {enemy_hp}")
    print(f"{'─'*70}")
    
    action = input("\n[A]ttack, [D]efend, [S]pell, or [R]un? ").strip().upper()
    
    if action == 'A':
        roll, total = roll_dice(20, character.attack_bonus)
        print(f"\n⚔️  You attack!")
        print(f"🎲 Rolled: {roll} + {character.attack_bonus} = {total}")
        
        if total >= enemy_ac:
            damage = random.randint(2, 6) + 2
            enemy_hp -= damage
            print(f"✅ HIT! {damage} damage!")
            
            if enemy_hp <= 0:
                print(f"\n💀 {enemy_name} defeated!")
                return 'victory', 0
        else:
            print(f"❌ MISS! (needed {enemy_ac}+)")
        
        if enemy_hp > 0:
            print(f"\n🔴 {enemy_name} counterattacks!")
            enemy_roll, enemy_total = roll_dice(20, 2)
            print(f"🎲 Enemy: {enemy_roll} + 2 = {enemy_total}")
            
            if enemy_total >= character.defense:
                damage = random.randint(1, 4) + 1
                character.hp -= damage
                print(f"💥 HIT! {damage} damage!")
                if character.hp <= 0:
                    return 'defeat', 0
            else:
                print(f"🛡️  MISS!")
    
    elif action == 'S':
        print(f"\n✨ Casting spell!")
        roll, total = roll_dice(20, character.attack_bonus + 1)
        print(f"🎲 Spell: {roll} + {character.attack_bonus + 1} = {total}")
        
        if total >= enemy_ac - 1:
            damage = random.randint(3, 8) + 2
            enemy_hp -= damage
            print(f"⚡ MAGICAL HIT! {damage} damage!")
            if enemy_hp <= 0:
                print(f"\n💀 {enemy_name} defeated!")
                return 'victory', 0
        else:
            print(f"❌ Spell fizzles!")
    
    elif action == 'D':
        print(f"\n🛡️  Defensive stance!")
        roll1, _ = roll_dice(20)
        roll2, _ = roll_dice(20)
        enemy_roll = min(roll1, roll2)
        enemy_total = enemy_roll + 2
        print(f"🔴 Enemy attacks (disadvantage)!")
        print(f"🎲 Rolled {roll1}, {roll2} → {enemy_roll}")
        
        if enemy_total >= character.defense + 2:
            damage = max(1, random.randint(1, 4))
            character.hp -= damage
            print(f"💥 {damage} damage!")
        else:
            print(f"🛡️  BLOCKED!")
    
    elif action == 'R':
        roll, total = roll_dice(20)
        print(f"\n🏃 Fleeing!")
        print(f"🎲 Rolled: {roll}")
        
        if total >= 10:
            print(f"✅ Escaped!")
            return 'fled', enemy_hp
        else:
            print(f"❌ Can't escape!")
            damage = random.randint(1, 3)
            character.hp -= damage
            print(f"💥 {damage} damage!")
    
    return 'continue', enemy_hp

# ============================================================================
# GAME FUNCTIONS
# ============================================================================

def create_character():
    print("\n" + "="*70)
    print("⚔️  CHARACTER CREATION")
    print("="*70)
    
    name = input("\nCharacter name? ").strip() or "Adventurer"
    
    print("\nClass:")
    print("1. Wizard")
    print("2. Fighter")
    print("3. Rogue")
    print("4. Cleric")
    
    classes = {'1': 'Wizard', '2': 'Fighter', '3': 'Rogue', '4': 'Cleric'}
    char_class = classes.get(input("Choice: ").strip(), 'Fighter')
    
    print("\nRace:")
    print("1. Human")
    print("2. Elf")
    print("3. Dwarf")
    print("4. Halfling")
    
    races = {'1': 'Human', '2': 'Elf', '3': 'Dwarf', '4': 'Halfling'}
    race = races.get(input("Choice: ").strip(), 'Human')
    
    return Character(name, char_class, race)

def show_menu(enemy_present=False):
    print("\n" + "─"*70)
    if enemy_present:
        print("⚔️  ENEMY PRESENT!")
        print("1. FIGHT")
        print("2. RUN")
    else:
        print("1. Explore")
        print("2. Next location")
        print("3. Search")
        print("4. Talk to NPC")
        print("5. Items")
        print("6. Rest")
        print("7. Stats")
        print("8. View campaign")
    print("0. Quit")
    print("─"*70)

# ============================================================================
# MAIN GAME
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🎮 UT LONGHORN DM - LLM-POWERED")
    print("="*70 + "\n")
    
    # Generate campaign
    generator = CampaignGenerator()
    campaign_content = generator.generate_campaign_content()
    
    # Create character
    game_state = GameState()
    game_state.campaign_content = campaign_content
    game_state.character = create_character()
    
    # Opening
    print("\n" + "="*70)
    print("🏰 YOUR ADVENTURE BEGINS")
    print("="*70)
    print(f"\n📖 Welcome, {game_state.character.name}!")
    print("   Your LLM has generated a unique campaign.\n")
    print(game_state.character)
    
    input("Press Enter...")
    
    # Main loop
    while game_state.character.hp > 0:
        print("\n" + "="*70)
        print(f"📍 Location {game_state.location_index + 1}")
        
        # Show LLM location
        loc_desc = campaign_content['locations'][game_state.location_index % len(campaign_content['locations'])]
        print(f"\n📖 {loc_desc}")
        
        print("="*70)
        
        show_menu(game_state.enemy_present)
        choice = input("\nChoice: ").strip()
        
        if choice == '0':
            print("\n👋 Thanks for playing!")
            break
        
        if game_state.enemy_present:
            if choice == '1':  # Fight
                enemy_desc = random.choice(campaign_content['enemies'])
                print(f"\n⚔️  Battle!")
                print(f"📖 {enemy_desc}\n")
                
                enemy_hp = random.randint(10, 15)
                enemy_ac = 11
                
                while game_state.character.hp > 0 and enemy_hp > 0:
                    result, new_hp = combat_round(
                        game_state.character, "Enemy", enemy_hp, enemy_ac
                    )
                    
                    enemy_hp = new_hp
                    
                    if result == 'victory':
                        xp, gold = 30, 10
                        game_state.character.xp += xp
                        game_state.character.gold += gold
                        print(f"\n✨ Victory! +{xp} XP, +{gold} gold")
                        game_state.enemy_present = False
                        break
                    elif result == 'defeat':
                        print(f"\n💀 GAME OVER")
                        return
                    elif result == 'fled':
                        game_state.enemy_present = False
                        break
            
            elif choice == '2':  # Run
                roll, total = roll_dice(20)
                print(f"\n🏃 Fleeing! Rolled: {roll}")
                if total >= 11:
                    print(f"✅ Escaped!")
                    game_state.enemy_present = False
                else:
                    print(f"❌ Can't escape!")
        
        else:
            if choice == '1':  # Explore
                print(f"\n🔍 Exploring...")
                
                # 30% encounter
                if random.random() < 0.3:
                    print(f"⚠️  Enemy appears!")
                    enemy_desc = random.choice(campaign_content['enemies'])
                    print(f"📖 {enemy_desc}")
                    game_state.enemy_present = True
                else:
                    print(f"   Nothing dangerous found.")
            
            elif choice == '2':  # Next location
                game_state.location_index = (game_state.location_index + 1) % len(campaign_content['locations'])
                print(f"\n🗺️  Moving...")
            
            elif choice == '3':  # Search
                roll, total = roll_dice(20)
                print(f"\n🔎 Searching... Rolled: {roll}")
                
                if total >= 12:
                    items = ["Healing Potion", "Magic Scroll", "5 Gold"]
                    found = random.choice(items)
                    print(f"✨ Found: {found}!")
                    if "Gold" in found:
                        game_state.character.gold += 5
                    else:
                        game_state.character.inventory.append(found)
                else:
                    print(f"Nothing found.")
            
            elif choice == '4':  # Talk
                npc = random.choice(campaign_content['npcs'])
                print(f"\n💬 You meet someone...")
                print(f"📖 {npc}")
            
            elif choice == '5':  # Items
                if not game_state.character.inventory:
                    print("\n❌ Empty inventory!")
                else:
                    print("\n🎒 Inventory:")
                    for i, item in enumerate(game_state.character.inventory, 1):
                        print(f"   {i}. {item}")
                    
                    try:
                        idx = int(input("\nUse (0=cancel): ")) - 1
                        if idx >= 0:
                            item = game_state.character.inventory.pop(idx)
                            if "Potion" in item:
                                heal = random.randint(2, 8) + 4
                                game_state.character.hp = min(game_state.character.hp + heal, game_state.character.max_hp)
                                print(f"\n✨ Healed {heal} HP!")
                    except:
                        pass
            
            elif choice == '6':  # Rest
                heal = random.randint(1, 6) + 2
                game_state.character.hp = min(game_state.character.hp + heal, game_state.character.max_hp)
                print(f"\n😴 Rested. +{heal} HP")
            
            elif choice == '7':  # Stats
                print(game_state.character)
            
            elif choice == '8':  # Campaign
                print("\n" + "="*70)
                print("📜 YOUR LLM-GENERATED CAMPAIGN")
                print("="*70)
                
                print("\n📍 Locations:")
                for i, loc in enumerate(campaign_content['locations'], 1):
                    print(f"   {i}. {loc}")
                
                print("\n👥 NPCs:")
                for i, npc in enumerate(campaign_content['npcs'], 1):
                    print(f"   {i}. {npc}")
                
                print("\n⚔️  Enemies:")
                for i, enemy in enumerate(campaign_content['enemies'], 1):
                    print(f"   {i}. {enemy}")
                
                print("\n📜 Quest Hooks:")
                for i, quest in enumerate(campaign_content['quests'], 1):
                    print(f"   {i}. {quest}")
                
                input("\nPress Enter...")
    
    print("\n" + "="*70)
    print("GAME OVER")
    print("="*70)
    print(game_state.character)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
